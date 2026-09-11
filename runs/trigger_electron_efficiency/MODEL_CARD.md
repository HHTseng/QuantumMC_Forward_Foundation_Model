# Trigger-electron efficiency model

The checkpoint approximates

$$
\widehat\epsilon_e(x_e)\simeq P(T=1\mid x_e).
$$

- denominator: `gen_pid = 11` (5,000,000 events);
- features: `log1p_gen_p, gen_theta, sin_gen_phi, cos_gen_phi, gen_vz`;
- label: `has_valid_trigger_electron`;
- selection: epoch 14, minimum validation BCE;
- test rows: 500,438;
- observed/predicted rate: 0.494381/0.492637;
- BCE/Brier/ECE: 0.228773/0.066687/0.003053.
