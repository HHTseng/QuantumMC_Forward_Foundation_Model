from __future__ import annotations

import copy
import math
import tempfile
import unittest
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import torch

from forwardfm_step1.config import load_config
from forwardfm_step1 import data_asinh_dp as data
from forwardfm_step1.model_split_dp import (
    ConditionalSplitMDN, SplitModelOutput, initialize_raw_momentum_from_control,
    split_mixture_nll, sample_split_standardized_residuals,
)
from forwardfm_step1.training_split_dp import make_loader, DeviceBatches
from forwardfm_step1.evaluation_split_dp import fit_gaussian_core
from sample_split_dp_asinh import sample_frame, load_checkpoint

ROOT = Path(__file__).resolve().parent.parent


class SplitResponseTests(unittest.TestCase):
    def test_transform_inverse_and_unchanged_other_targets(self):
        physical = np.array([[-10,.01,-.02,.03],[0,0,0,0],[.02,.1,.2,.3],[10,1,2,3]],dtype=np.float32)
        transformed = data.transform_targets_for_model(physical)
        np.testing.assert_allclose(data.inverse_targets_from_model(transformed),physical,atol=1e-6,rtol=2e-6)
        np.testing.assert_array_equal(transformed[:,1:],physical[:,1:])
        self.assertAlmostEqual(float(transformed[2,0]),np.arcsinh(1),places=6)

    def test_data_alignment_audit_and_independent_split_seed(self):
        with tempfile.TemporaryDirectory() as temp:
            rows=[]
            for event in range(180):
                for index,pid in enumerate(data.SPECIES):
                    p=.2+event/20
                    rows.append(dict(source_file_id=1,event_id=event,mcindex=index,gen_pid=pid,
                        gen_p=p,gen_theta=.2,gen_phi=event/50,gen_vz=-3.,rec_theta=.21,
                        rec_pid=pid,rec_beta=p/math.sqrt(p*p+data.GEN_MASS_GEV[pid]**2)-.003,
                        rec_detector_region='FD',match_reciprocal=True,
                        usable_for_hadron_response_training=True,delta_p=-.01,
                        delta_theta=.01,delta_phi=.002))
            frame=pd.DataFrame(rows)
            con=duckdb.connect(); con.register('input_frame',frame)
            path=Path(temp)/'teacher.parquet'
            con.execute(f"COPY input_frame TO '{path}' (FORMAT PARQUET)"); con.close()
            base=load_config(ROOT/'configs/gpu_split_dp_asinh.yaml')
            base['data']['parquet_glob']=str(path)
            a,afs,ats,av,aa=data.load_all_splits(base)
            raw=copy.deepcopy(base); raw['project']['seed']+=1
            raw['data']['include_raw_generated_momentum']=True
            b,bfs,bts,bv,ba=data.load_all_splits(raw)
            self.assertEqual(av,bv)
            self.assertEqual(aa['particle_order_sha256'],ba['particle_order_sha256'])
            self.assertEqual(aa['event_split_overlap_count'],0)
            self.assertEqual(aa['duplicate_composite_particle_keys'],0)
            for name in a:
                np.testing.assert_array_equal(a[name].event_keys,b[name].event_keys)
                np.testing.assert_array_equal(a[name].targets,b[name].targets)
                np.testing.assert_allclose(a[name].continuous,b[name].continuous[:,:5],atol=1e-6)
            leaked=copy.deepcopy(a); leaked['validation'].event_keys[0]=a['train'].event_keys[0]
            with self.assertRaises(AssertionError): data.assert_event_disjoint(leaked)

    def test_nested_initial_function_and_trainable_raw_column(self):
        torch.manual_seed(4)
        base=ConditionalSplitMDN(5,3,6,hidden_width=16,hidden_layers=2,dropout=0)
        raw=ConditionalSplitMDN(6,3,6,hidden_width=16,hidden_layers=2,dropout=0)
        initialize_raw_momentum_from_control(base,raw)
        x=torch.randn(20,6); species=torch.arange(20)%3
        a,b=base(x[:,:5],species),raw(x,species)
        for name in vars(a):
            torch.testing.assert_close(getattr(a,name),getattr(b,name),atol=2e-6,rtol=2e-5)
        self.assertEqual(float(raw.backbone[0].weight[:,5].detach().abs().sum()),0)
        split_mixture_nll(b,torch.randn(20,4)).backward()
        self.assertGreater(float(raw.backbone[0].weight.grad[:,5].abs().sum()),0)

    def test_preloaded_batch_sequence_matches_reference_for_three_epochs(self):
        n=31
        split=data.PreparedSplit('train',np.arange(n),np.arange(n*5,dtype=np.float32).reshape(n,5),
            np.arange(n)%3,np.zeros((n,4),dtype=np.float32),np.arange(n)%2,np.full(n,211))
        reference=make_loader(split,8,True,17,0)
        fast=DeviceBatches(make_loader(split,8,True,17,0),torch.device('cpu'))
        for _ in range(3):
            a,b=list(reference),list(fast)
            self.assertEqual(len(a),len(b))
            for left,right in zip(a,b):
                for x,y in zip(left,right): torch.testing.assert_close(x,y,atol=0,rtol=0)

    def test_factorized_normal_nll(self):
        output=SplitModelOutput(torch.zeros(4,3),torch.zeros(4,3),torch.zeros(4,3),
            torch.zeros(4,3),torch.zeros(4,3,3),torch.zeros(4,3,3),torch.zeros(4,2))
        self.assertAlmostEqual(float(split_mixture_nll(output,torch.zeros(4,4))),2*math.log(2*math.pi),places=5)

    def test_checkpoint_roundtrip_includes_nonlinear_inverse(self):
        with tempfile.TemporaryDirectory() as temp:
            torch.manual_seed(9)
            model=ConditionalSplitMDN(6,3,4,hidden_width=12,hidden_layers=2,dropout=0).eval()
            ck={'format_version':2,'model_type':'split_delta_p_asinh','architecture':model.architecture_dict(),
                'model_state':model.state_dict(),'feature_names':[*data.CONTINUOUS_FEATURES,'gen_p'],
                'target_names':list(data.TARGET_COLUMNS),'target_transform':dict(data.TARGET_TRANSFORM),
                'particle_masses_gev':{str(k):v for k,v in data.GEN_MASS_GEV.items()},
                'species_pids':list(data.SPECIES),'rec_pid_vocabulary':list(data.SPECIES),
                'feature_scaler':{'mean':[0.]*6,'scale':[1.]*6},
                'target_scaler':{'mean':[.4,0.,0.,0.],'scale':[.5,.001,.001,.001]}}
            path=Path(temp)/'model.pt'; torch.save(ck,path)
            frame=pd.DataFrame(dict(gen_pid=[-211,211,2212],gen_p=[1.,2.,3.],gen_theta=[.2,.3,.4],gen_phi=[0.,1.,2.]))
            sampled=sample_frame(path,frame,torch.device('cpu'),20261003)
            output=model(torch.from_numpy(data.build_features(frame,tuple(ck['feature_names']))),torch.arange(3))
            draws=sample_split_standardized_residuals(output,torch.Generator().manual_seed(20261003)).detach().numpy()
            expected=data.inverse_targets_from_model(data.Standardizer.from_dict(ck['target_scaler']).inverse(draws))
            np.testing.assert_allclose(sampled.sampled_delta_p,expected[:,0],rtol=1e-6)
            self.assertTrue(np.isfinite(sampled.sampled_rec_beta).all())
            ck['target_scaler']['mean'][0]=20.
            torch.save(ck,path)
            extreme=sample_frame(path,frame,torch.device('cpu'),20261003)
            self.assertTrue(extreme.sample_abs_delta_p_gt_10GeV.all())
            self.assertGreater(float(extreme.sampled_delta_p.min()),10.)
            del ck['target_transform']; torch.save(ck,path)
            with self.assertRaises(ValueError): load_checkpoint(path,torch.device('cpu'))

    def test_gaussian_fit_measures_known_core(self):
        fit=fit_gaussian_core(np.random.default_rng(7).normal(-.004,.012,100000))
        self.assertIsNotNone(fit)
        self.assertLess(abs(fit['mu']+4),.2)
        self.assertLess(abs(fit['sigma']-12),.3)


if __name__=='__main__':
    unittest.main()
