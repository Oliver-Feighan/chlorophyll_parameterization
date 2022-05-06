%nproc=24
%mem=75GB
#p cam-b3lyp/Def2SVP td=(nstates=5)

step_1701_chromophore_17 TDDFT with cam-b3lyp functional

0 1
Mg   29.219   58.717   41.571
C   26.262   57.076   40.846
C   30.882   56.077   39.968
C   31.790   60.698   41.362
C   27.153   61.487   42.615
N   28.630   56.796   40.580
C   27.395   56.403   40.306
C   27.307   55.156   39.539
C   28.794   54.675   39.576
C   29.512   55.916   40.043
C   28.998   53.531   40.605
C   26.737   55.433   38.034
C   26.283   54.251   37.215
C   25.730   54.608   35.795
O   24.527   54.612   35.552
O   26.729   54.912   34.863
N   31.151   58.371   40.974
C   31.682   57.230   40.340
C   33.074   57.386   40.136
C   33.333   58.747   40.350
C   32.041   59.415   40.912
C   34.025   56.319   39.681
C   34.597   59.456   40.066
O   35.574   58.821   39.662
C   34.762   60.916   40.079
N   29.407   60.895   41.858
C   30.653   61.402   41.865
C   30.613   62.841   42.351
C   29.187   63.069   42.876
C   28.474   61.775   42.325
C   31.688   63.336   43.284
C   28.565   64.476   42.426
C   28.076   65.459   43.545
N   27.075   59.190   41.846
C   26.431   60.309   42.323
C   25.019   60.068   42.365
C   24.893   58.842   41.754
C   26.193   58.369   41.518
C   23.977   60.917   42.995
C   23.957   57.770   41.456
O   22.718   57.755   41.475
C   24.835   56.606   40.769
C   24.541   55.397   41.526
O   24.576   55.279   42.726
O   23.941   54.381   40.765
C   23.551   53.126   41.548
C   26.440   55.099   33.423
C   27.247   56.194   32.895
C   27.226   56.636   31.602
C   26.606   55.834   30.452
C   28.045   57.830   31.042
C   29.568   57.700   31.157
C   30.294   56.907   30.008
C   31.080   55.667   30.543
C   30.144   54.488   30.792
C   32.385   55.319   29.695
C   33.675   54.960   30.531
C   35.005   55.604   30.040
C   36.096   54.569   29.587
C   36.956   54.161   30.797
C   36.974   55.062   28.321
C   36.959   54.011   27.232
C   36.412   54.562   25.831
C   37.400   54.323   24.621
C   38.372   55.501   24.422
C   36.661   54.117   23.289
H   31.346   55.211   39.492
H   32.654   61.358   41.260
H   26.624   62.280   43.149
H   26.643   54.424   39.999
H   29.136   54.324   38.603
H   28.223   53.712   41.350
H   29.980   53.669   41.057
H   29.091   52.515   40.221
H   27.577   55.830   37.464
H   25.972   56.209   38.066
H   25.439   53.871   37.790
H   27.068   53.498   37.151
H   33.391   55.432   39.681
H   34.688   56.188   40.536
H   34.454   56.469   38.690
H   34.047   61.393   39.408
H   35.708   61.344   39.746
H   34.737   61.329   41.088
H   30.817   63.470   41.485
H   29.226   63.094   43.965
H   32.563   62.687   43.315
H   31.436   63.332   44.344
H   32.024   64.320   42.955
H   27.770   64.378   41.687
H   29.334   64.963   41.827
H   27.014   65.656   43.394
H   28.621   66.402   43.541
H   28.170   64.938   44.498
H   24.133   61.993   42.921
H   23.981   60.823   44.081
H   23.039   60.575   42.559
H   24.557   56.496   39.721
H   24.087   53.014   42.491
H   23.719   52.268   40.897
H   22.484   53.276   41.709
H   25.405   55.442   33.436
H   26.578   54.102   33.007
H   27.687   56.762   33.714
H   27.402   55.252   29.987
H   26.163   56.361   29.607
H   25.806   55.205   30.843
H   27.717   58.702   31.608
H   27.903   58.025   29.979
H   29.846   57.175   32.071
H   30.046   58.680   31.143
H   30.972   57.517   29.411
H   29.496   56.609   29.328
H   31.505   55.858   31.529
H   29.589   54.543   31.729
H   30.774   53.599   30.836
H   29.458   54.410   29.949
H   32.610   56.191   29.082
H   32.140   54.438   29.101
H   33.716   53.871   30.517
H   33.435   55.257   31.552
H   35.363   56.270   30.825
H   34.862   56.330   29.239
H   35.600   53.642   29.298
H   37.239   55.095   31.284
H   37.864   53.639   30.493
H   36.410   53.518   31.486
H   38.023   55.207   28.580
H   36.688   56.029   27.908
H   36.216   53.229   27.384
H   37.977   53.623   27.182
H   36.034   55.584   25.861
H   35.595   53.870   25.626
H   38.013   53.474   24.924
H   39.313   55.285   24.928
H   37.836   56.389   24.755
H   38.712   55.752   23.417
H   36.875   53.100   22.959
H   37.131   54.900   22.693
H   35.573   54.155   23.234

