%nproc=24
%mem=75GB
#p blyp/Def2SVP td=(nstates=5)

step_1151_chromophore_22 TDDFT with blyp functional

0 1
Mg   8.975   48.595   24.898
C   6.692   48.287   27.489
C   11.361   49.578   27.117
C   11.238   48.580   22.413
C   6.382   47.978   22.594
N   8.948   49.081   27.065
C   7.918   48.772   27.938
C   8.306   49.217   29.302
C   9.712   49.881   29.107
C   10.057   49.530   27.659
C   9.783   51.368   29.424
C   8.072   48.130   30.395
C   8.887   48.123   31.679
C   8.317   47.845   33.052
O   7.836   48.704   33.773
O   8.328   46.482   33.355
N   11.061   48.863   24.775
C   11.857   49.298   25.817
C   13.244   49.355   25.334
C   13.241   48.960   23.964
C   11.785   48.765   23.669
C   14.328   49.796   26.242
C   14.363   48.713   23.025
O   14.113   48.326   21.853
C   15.795   48.687   23.441
N   8.804   48.444   22.753
C   9.910   48.494   21.955
C   9.541   48.350   20.443
C   7.961   47.982   20.583
C   7.690   48.109   22.122
C   9.853   49.515   19.489
C   7.433   46.644   19.986
C   6.180   46.779   19.107
N   6.964   48.378   24.919
C   6.028   48.031   23.935
C   4.736   47.838   24.540
C   4.922   47.846   25.921
C   6.304   48.145   26.084
C   3.515   47.701   23.790
C   4.309   47.797   27.230
O   3.143   47.562   27.513
C   5.460   47.879   28.317
C   5.766   46.599   28.905
O   6.401   45.696   28.452
O   5.052   46.584   30.081
C   5.147   45.432   30.968
C   7.477   46.056   34.531
C   8.229   46.000   35.814
C   7.827   46.093   37.088
C   6.364   46.169   37.387
C   8.882   46.242   38.196
C   9.403   47.662   38.598
C   10.858   47.759   39.107
C   11.165   46.950   40.408
C   10.972   47.921   41.513
C   12.564   46.320   40.390
C   12.616   44.765   40.441
C   12.744   44.290   41.916
C   14.042   43.754   42.454
C   14.067   44.242   43.975
C   14.175   42.199   42.367
C   15.645   41.756   42.443
C   15.969   41.096   41.088
C   15.834   39.566   41.060
C   16.898   38.877   40.086
C   14.464   39.177   40.527
H   12.179   49.827   27.796
H   11.933   48.556   21.571
H   5.715   47.603   21.815
H   7.592   49.964   29.649
H   10.479   49.332   29.653
H   8.782   51.787   29.331
H   10.303   51.879   28.614
H   10.224   51.504   30.412
H   8.376   47.187   29.940
H   7.030   48.259   30.687
H   9.131   49.173   31.841
H   9.790   47.515   31.618
H   13.946   50.300   27.129
H   14.877   50.508   25.626
H   14.932   48.923   26.491
H   16.049   47.722   23.879
H   16.014   49.404   24.232
H   16.422   48.598   22.553
H   10.026   47.480   20.000
H   7.442   48.851   20.178
H   8.961   49.851   18.962
H   10.674   49.251   18.822
H   10.097   50.340   20.159
H   7.258   46.013   20.858
H   8.246   46.205   19.407
H   5.922   47.817   18.897
H   5.350   46.327   19.650
H   6.334   46.279   18.150
H   3.691   46.949   23.022
H   3.273   48.698   23.422
H   2.764   47.338   24.492
H   5.188   48.666   29.020
H   5.815   44.614   30.699
H   4.113   45.093   30.908
H   5.388   45.822   31.957
H   7.102   45.036   34.446
H   6.590   46.685   34.615
H   9.278   46.096   35.535
H   6.109   47.181   37.700
H   6.054   45.510   38.198
H   5.743   45.936   36.522
H   9.710   45.577   37.947
H   8.333   45.832   39.043
H   8.751   48.053   39.378
H   9.412   48.268   37.692
H   11.081   48.821   39.005
H   11.436   47.221   38.356
H   10.379   46.208   40.545
H   11.105   48.997   41.395
H   11.550   47.571   42.367
H   9.910   47.826   41.739
H   13.147   46.624   41.260
H   13.235   46.724   39.631
H   13.438   44.461   39.794
H   11.684   44.481   39.952
H   11.989   43.515   42.045
H   12.239   45.041   42.523
H   14.870   44.265   41.961
H   15.115   44.365   44.247
H   13.518   43.633   44.695
H   13.734   45.276   44.065
H   13.730   41.883   41.423
H   13.676   41.767   43.234
H   15.668   41.072   43.291
H   16.234   42.671   42.507
H   17.019   41.352   40.940
H   15.430   41.560   40.262
H   15.860   39.013   41.999
H   17.486   39.659   39.605
H   16.376   38.293   39.329
H   17.542   38.254   40.706
H   13.776   39.501   41.308
H   14.302   38.122   40.304
H   14.235   39.789   39.655
