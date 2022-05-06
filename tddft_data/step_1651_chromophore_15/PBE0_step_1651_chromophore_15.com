%nproc=24
%mem=75GB
#p PBE1PBE/Def2SVP td=(nstates=5)

step_1651_chromophore_15 TDDFT with PBE1PBE functional

0 1
Mg   46.914   35.248   28.262
C   45.631   33.049   30.723
C   47.071   37.725   30.717
C   47.622   37.475   25.928
C   46.181   32.863   25.942
N   46.512   35.402   30.458
C   46.003   34.332   31.165
C   45.951   34.764   32.666
C   45.948   36.298   32.584
C   46.600   36.470   31.213
C   44.598   37.126   32.805
C   47.243   34.170   33.335
C   47.418   34.290   34.901
C   46.318   34.950   35.748
O   45.331   34.368   36.140
O   46.636   36.281   35.820
N   47.449   37.337   28.327
C   47.574   38.148   29.386
C   48.140   39.400   29.014
C   48.252   39.440   27.611
C   47.814   38.052   27.237
C   48.310   40.566   29.939
C   48.631   40.580   26.664
O   48.597   40.525   25.450
C   49.177   41.918   27.241
N   46.781   35.127   26.259
C   47.255   36.168   25.481
C   47.476   35.702   24.036
C   47.004   34.246   24.043
C   46.648   34.046   25.528
C   46.865   36.687   22.931
C   48.107   33.151   23.670
C   47.586   31.952   22.802
N   46.075   33.304   28.305
C   45.867   32.449   27.279
C   45.303   31.218   27.753
C   45.083   31.394   29.141
C   45.722   32.660   29.403
C   45.017   29.959   26.989
C   44.479   30.831   30.340
O   43.874   29.796   30.584
C   44.749   31.985   31.386
C   45.210   31.347   32.597
O   46.357   30.895   32.773
O   44.136   31.389   33.510
C   44.386   30.883   34.818
C   45.838   36.938   36.845
C   46.833   37.813   37.575
C   46.620   38.360   38.794
C   45.211   38.365   39.482
C   47.602   39.456   39.228
C   47.407   40.789   38.522
C   48.111   41.867   39.303
C   47.811   43.274   38.662
C   48.885   43.689   37.665
C   47.606   44.423   39.800
C   46.128   44.923   40.071
C   45.985   46.464   40.103
C   45.736   47.139   38.658
C   46.699   48.305   38.466
C   44.277   47.667   38.539
C   43.205   46.633   38.125
C   42.214   46.266   39.314
C   41.435   45.013   38.999
C   40.724   45.015   37.561
C   40.326   44.753   40.073
H   47.077   38.504   31.483
H   47.988   38.213   25.211
H   46.046   32.097   25.176
H   45.056   34.285   33.063
H   46.643   36.556   33.383
H   44.588   37.906   32.044
H   44.632   37.512   33.823
H   43.735   36.468   32.707
H   48.118   34.693   32.948
H   47.314   33.108   33.099
H   48.362   34.804   35.082
H   47.613   33.289   35.286
H   49.323   40.967   29.950
H   48.094   40.430   30.998
H   47.587   41.339   29.676
H   50.070   41.591   27.773
H   48.507   42.220   28.046
H   49.289   42.792   26.599
H   48.539   35.687   23.795
H   46.116   34.037   23.445
H   47.630   37.088   22.267
H   46.329   37.477   23.457
H   46.073   36.185   22.376
H   48.487   32.861   24.649
H   48.827   33.718   23.079
H   47.559   32.163   21.733
H   46.566   31.654   23.045
H   48.172   31.038   22.901
H   44.180   29.478   27.495
H   45.838   29.249   26.888
H   44.548   30.255   26.050
H   43.755   32.363   31.624
H   45.082   30.048   34.897
H   43.436   30.731   35.331
H   44.832   31.660   35.438
H   45.598   36.229   37.638
H   44.958   37.411   36.410
H   47.717   38.138   37.027
H   45.176   37.444   40.062
H   44.441   38.401   38.710
H   44.983   39.173   40.178
H   48.556   38.962   39.044
H   47.406   39.544   40.296
H   46.384   40.954   38.184
H   47.958   40.692   37.586
H   49.118   41.549   39.576
H   47.545   41.868   40.235
H   46.851   43.159   38.160
H   49.564   44.432   38.084
H   48.383   44.053   36.769
H   49.501   42.893   37.247
H   48.151   45.316   39.494
H   48.020   44.055   40.739
H   45.860   44.593   41.074
H   45.380   44.480   39.413
H   46.895   46.826   40.581
H   45.129   46.636   40.755
H   45.805   46.543   37.748
H   46.136   49.222   38.646
H   47.047   48.265   37.433
H   47.564   48.248   39.126
H   44.249   48.600   37.977
H   43.909   48.066   39.485
H   43.523   45.695   37.670
H   42.593   47.184   37.410
H   41.388   46.974   39.378
H   42.776   46.144   40.240
H   42.151   44.194   39.071
H   41.323   44.667   36.719
H   40.529   46.067   37.352
H   39.763   44.512   37.459
H   40.471   45.408   40.932
H   40.318   43.710   40.388
H   39.366   45.083   39.677

