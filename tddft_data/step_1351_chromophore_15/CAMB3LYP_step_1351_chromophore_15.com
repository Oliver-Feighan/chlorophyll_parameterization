%nproc=24
%mem=75GB
#p cam-b3lyp/Def2SVP td=(nstates=5)

step_1351_chromophore_15 TDDFT with cam-b3lyp functional

0 1
Mg   46.127   34.810   28.249
C   44.352   32.733   30.641
C   46.145   37.190   30.872
C   47.111   37.095   25.964
C   45.417   32.481   25.984
N   45.187   34.911   30.390
C   44.783   33.937   31.207
C   44.788   34.322   32.675
C   44.844   35.871   32.553
C   45.395   36.038   31.143
C   43.474   36.547   32.822
C   46.005   33.646   33.423
C   46.309   33.842   35.001
C   45.469   34.787   35.800
O   44.331   34.528   36.206
O   46.140   35.953   36.182
N   46.793   36.777   28.430
C   46.698   37.594   29.576
C   47.381   38.821   29.301
C   47.735   38.788   27.845
C   47.195   37.500   27.326
C   47.656   39.862   30.315
C   48.361   39.796   26.974
O   48.556   39.588   25.766
C   48.719   41.113   27.529
N   46.218   34.775   26.301
C   46.648   35.825   25.507
C   46.651   35.443   24.018
C   46.329   33.933   24.069
C   45.992   33.653   25.514
C   45.809   36.283   23.100
C   47.326   33.011   23.319
C   46.656   32.082   22.264
N   45.231   32.881   28.269
C   45.074   32.029   27.207
C   44.320   30.795   27.665
C   44.085   31.020   29.028
C   44.577   32.355   29.348
C   43.767   29.659   26.851
C   43.417   30.503   30.147
O   42.862   29.413   30.270
C   43.394   31.659   31.224
C   43.648   31.124   32.548
O   44.500   30.291   32.878
O   42.861   31.799   33.382
C   43.106   31.459   34.790
C   45.368   36.734   37.210
C   46.224   37.951   37.619
C   45.939   38.906   38.532
C   44.771   38.743   39.491
C   46.966   40.034   38.901
C   47.258   41.037   37.775
C   48.748   41.578   37.898
C   48.836   43.127   37.816
C   50.124   43.623   37.193
C   48.645   43.661   39.262
C   47.189   44.178   39.467
C   47.252   45.762   39.628
C   46.774   46.465   38.347
C   47.928   47.250   37.689
C   45.535   47.266   38.479
C   44.181   46.465   38.282
C   43.311   46.499   39.603
C   41.873   46.994   39.387
C   41.604   48.114   40.399
C   40.963   45.742   39.533
H   46.255   37.855   31.731
H   47.214   37.776   25.117
H   45.242   31.680   25.262
H   43.892   34.033   33.223
H   45.561   36.233   33.290
H   43.530   36.948   33.834
H   42.755   35.727   32.843
H   43.238   37.287   32.058
H   46.915   33.913   32.887
H   45.855   32.573   33.305
H   47.349   34.100   35.200
H   46.084   32.898   35.497
H   46.945   40.667   30.134
H   48.653   40.247   30.097
H   47.694   39.413   31.308
H   49.673   41.092   28.057
H   47.914   41.435   28.189
H   48.862   41.864   26.752
H   47.697   35.578   23.742
H   45.410   33.935   23.482
H   45.590   37.245   23.564
H   44.969   35.713   22.702
H   46.472   36.535   22.273
H   47.945   32.475   24.039
H   48.077   33.571   22.762
H   46.945   32.440   21.276
H   45.574   32.025   22.378
H   47.135   31.108   22.370
H   42.808   29.381   27.288
H   44.442   28.808   26.758
H   43.629   30.201   25.915
H   42.340   31.937   31.202
H   42.772   30.516   35.223
H   42.546   32.301   35.197
H   44.153   31.387   35.085
H   45.233   36.127   38.105
H   44.391   37.075   36.866
H   47.134   38.139   37.049
H   45.048   39.113   40.478
H   44.451   37.704   39.575
H   43.965   39.381   39.130
H   47.804   39.593   39.442
H   46.459   40.533   39.727
H   46.648   41.934   37.885
H   47.193   40.532   36.811
H   49.181   41.261   36.949
H   49.194   41.135   38.789
H   48.047   43.639   37.267
H   50.718   43.991   38.029
H   49.778   44.349   36.457
H   50.596   42.775   36.695
H   49.450   44.340   39.546
H   48.718   42.881   40.019
H   46.838   43.708   40.386
H   46.537   43.963   38.620
H   48.236   46.184   39.828
H   46.535   45.785   40.449
H   46.584   45.718   37.576
H   47.995   46.945   36.645
H   48.932   47.097   38.083
H   47.833   48.321   37.871
H   45.552   47.944   37.626
H   45.633   47.942   39.329
H   44.414   45.414   38.112
H   43.779   46.867   37.352
H   43.676   47.030   40.482
H   43.319   45.491   40.017
H   41.684   47.453   38.417
H   41.104   48.869   39.793
H   42.490   48.500   40.902
H   40.894   47.917   41.202
H   39.941   46.069   39.724
H   41.274   45.028   40.295
H   40.879   45.176   38.605

