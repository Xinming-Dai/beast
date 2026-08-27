# Plot notes

## Example Encoding Results
```text
'ecb5520d-1358-434c-95ec-93687ecd1396'
  'rrr'
    'gt': array(shape=(77, 60, 585), dtype=float64) min=0 max=9 mean=0.187229
    'pred': array(shape=(77, 60, 585), dtype=float64) min=0.00332726 max=1.81146 mean=0.187312
    'norm_gt': array(shape=(77, 60, 585), dtype=float64) min=-4.80411 max=70.8687 mean=0.00295457
    'norm_pred': array(shape=(77, 60, 585), dtype=float64) min=-0.0340919 max=0.0360399 mean=-1.08041e-05
    'mean_X': array(shape=(60, 200), dtype=float32) min=-0.162059 max=0.201172 mean=0.00182288
    'std_X': array(shape=(60, 200), dtype=float32) min=0.0455419 max=0.158371 mean=0.0882183
    'mean_y': array(shape=(60, 585), dtype=float64) min=0.00344467 max=1.80372 mean=0.187315
    'std_y': array(shape=(60, 585), dtype=float64) min=0.0215612 max=1.29795 mean=0.172782
    'bps': 0.0477281
    'r2': -0.00357442
    'eid': 'ecb5520d-1358-434c-95ec-93687ecd1396'
  'cnn'
    'gt': array(shape=(77, 60, 585), dtype=float64) min=0 max=9 mean=0.187229
    'pred': array(shape=(77, 60, 585), dtype=float64) min=0.001 max=2.07895 mean=0.187498
    'norm_gt': array(shape=(77, 60, 585), dtype=float32) min=-3.10903 max=99.4643 mean=0.00402942
    'norm_pred': array(shape=(77, 60, 585), dtype=float32) min=-0.166666 max=0.147458 mean=0.000978839
    'mean_X': array(shape=(60, 200), dtype=float32) min=-0.162059 max=0.201172 mean=0.00182288
    'std_X': array(shape=(60, 200), dtype=float32) min=0.0455419 max=0.158371 mean=0.0882183
    'mean_y': array(shape=(60, 585), dtype=float64) min=0.0016034 max=2.02888 mean=0.187315
    'std_y': array(shape=(60, 585), dtype=float64) min=0.0158996 max=1.64143 mean=0.230934
    'bps': 0.0498138
    'r2': -0.00283335
    'eid': 'ecb5520d-1358-434c-95ec-93687ecd1396'
```

[num_neural_trials, num_neural_bins, num_neurons]. 60 is 1/60s.
- 'gt' is the ground truth rate.
- 'pred' is the predicted rate.
- `bits_per_spike` uses `pred` and `gt` to compute the bits per spike.

## bps_bar_plot_figure3
