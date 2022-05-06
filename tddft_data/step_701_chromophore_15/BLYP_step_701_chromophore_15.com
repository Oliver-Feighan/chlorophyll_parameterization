%nproc=24
%mem=75GB
#p blyp/Def2SVP td=(nstates=5)

step_701_chromophore_15 TDDFT with blyp functional

0 1
Mg   47.215   35.024   28.033
C   45.482   32.988   30.228
C   47.349   37.529   30.479
C   48.216   37.129   25.639
C   46.461   32.674   25.394
N   46.544   35.259   30.073
C   46.053   34.164   30.795
C   46.242   34.436   32.317
C   46.314   36.011   32.311
C   46.687   36.353   30.851
C   45.187   36.879   32.889
C   47.505   33.669   32.826
C   47.630   33.525   34.317
C   46.675   34.341   35.239
O   45.546   33.988   35.579
O   47.276   35.473   35.799
N   47.906   36.953   28.091
C   47.824   37.802   29.213
C   48.569   39.069   28.834
C   48.767   39.054   27.420
C   48.271   37.697   26.987
C   49.085   40.283   29.848
C   49.283   40.103   26.475
O   49.317   39.908   25.280
C   49.579   41.537   26.864
N   47.410   34.842   25.828
C   47.870   35.869   25.092
C   47.856   35.547   23.620
C   47.638   34.038   23.596
C   47.017   33.872   24.983
C   46.781   36.378   22.905
C   48.880   33.122   23.435
C   48.589   31.750   22.873
N   46.090   33.272   27.819
C   45.985   32.416   26.750
C   45.422   31.123   27.226
C   45.126   31.336   28.582
C   45.513   32.653   28.859
C   45.172   29.842   26.403
C   44.513   30.802   29.787
O   44.130   29.657   30.007
C   44.494   31.872   30.806
C   45.093   31.319   32.026
O   46.131   30.716   32.121
O   44.288   31.653   33.088
C   44.678   31.074   34.442
C   46.459   36.167   36.820
C   47.337   37.090   37.645
C   46.951   37.902   38.655
C   45.596   37.969   39.230
C   47.995   38.761   39.248
C   48.338   40.056   38.424
C   49.548   40.826   38.869
C   49.277   42.338   38.945
C   50.555   43.074   39.396
C   48.126   42.711   40.000
C   46.879   43.391   39.338
C   46.449   44.560   40.180
C   46.708   45.859   39.475
C   47.921   46.594   39.947
C   45.462   46.816   39.686
C   44.252   46.550   38.725
C   43.140   45.666   39.409
C   41.818   46.468   39.429
C   41.664   47.162   40.805
C   40.610   45.547   39.131
H   47.473   38.321   31.221
H   48.458   37.936   24.944
H   46.251   31.941   24.612
H   45.346   34.097   32.835
H   47.156   36.239   32.963
H   44.352   36.238   33.172
H   44.880   37.535   32.075
H   45.564   37.514   33.691
H   48.457   34.071   32.480
H   47.603   32.658   32.432
H   48.626   33.805   34.658
H   47.713   32.491   34.652
H   50.158   40.471   29.816
H   48.977   39.907   30.866
H   48.449   41.124   29.574
H   50.356   41.618   27.624
H   48.591   41.875   27.175
H   49.922   42.130   26.016
H   48.811   35.773   23.146
H   46.900   33.858   22.814
H   47.277   36.841   22.052
H   46.402   37.301   23.343
H   45.943   35.836   22.465
H   49.365   32.994   24.403
H   49.502   33.586   22.670
H   47.561   31.387   22.876
H   49.050   30.961   23.467
H   48.986   31.649   21.863
H   45.660   29.892   25.430
H   44.108   29.621   26.317
H   45.603   28.989   26.928
H   43.495   32.229   31.057
H   44.160   31.577   35.259
H   45.755   31.185   34.573
H   44.409   30.031   34.608
H   46.114   35.509   37.618
H   45.579   36.668   36.417
H   48.368   37.229   37.319
H   45.403   39.031   39.382
H   45.479   37.481   40.197
H   44.925   37.635   38.439
H   48.886   38.135   39.248
H   47.766   39.141   40.244
H   47.410   40.628   38.425
H   48.450   39.691   37.403
H   50.301   40.738   38.085
H   50.044   40.375   39.729
H   49.056   42.750   37.961
H   50.363   44.082   39.763
H   51.254   43.125   38.561
H   51.031   42.541   40.219
H   48.651   43.320   40.736
H   47.757   41.851   40.560
H   46.058   42.687   39.207
H   47.212   43.660   38.336
H   46.951   44.608   41.146
H   45.383   44.423   40.364
H   46.786   45.628   38.412
H   48.512   45.928   40.576
H   47.646   47.445   40.570
H   48.586   46.780   39.103
H   45.705   47.870   39.554
H   45.081   46.829   40.707
H   44.485   46.022   37.800
H   43.929   47.531   38.378
H   43.317   45.408   40.453
H   43.050   44.735   38.851
H   41.851   47.132   38.564
H   40.941   46.574   41.371
H   41.327   48.196   40.731
H   42.564   47.053   41.410
H   40.598   44.673   39.782
H   40.621   45.162   38.111
H   39.685   46.111   39.248

