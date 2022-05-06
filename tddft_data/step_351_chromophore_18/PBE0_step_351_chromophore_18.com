%nproc=24
%mem=75GB
#p PBE1PBE/Def2SVP td=(nstates=5)

step_351_chromophore_18 TDDFT with PBE1PBE functional

0 1
Mg   35.029   49.442   25.472
C   34.930   47.587   28.598
C   33.619   51.933   27.154
C   34.636   50.893   22.472
C   35.557   46.341   23.875
N   34.327   49.700   27.720
C   34.453   48.912   28.771
C   33.887   49.584   30.044
C   33.475   51.024   29.487
C   33.806   50.907   28.037
C   32.047   51.363   29.728
C   34.927   49.790   31.147
C   34.671   49.171   32.504
C   33.269   49.275   33.136
O   32.666   48.348   33.593
O   32.849   50.588   33.142
N   34.439   51.256   24.850
C   33.953   52.194   25.738
C   33.873   53.447   24.997
C   34.099   53.145   23.675
C   34.382   51.740   23.599
C   33.586   54.739   25.708
C   34.155   54.113   22.494
O   34.470   53.860   21.356
C   33.896   55.569   22.737
N   35.159   48.759   23.568
C   34.889   49.524   22.479
C   34.794   48.733   21.148
C   35.472   47.384   21.598
C   35.433   47.493   23.118
C   33.409   48.660   20.520
C   36.836   46.960   21.013
C   38.186   47.309   21.716
N   35.232   47.368   26.047
C   35.530   46.262   25.256
C   35.583   45.117   26.120
C   35.416   45.587   27.435
C   35.271   47.020   27.285
C   36.001   43.731   25.728
C   35.388   45.210   28.832
O   35.609   44.111   29.350
C   35.107   46.480   29.657
C   33.834   46.116   30.286
O   32.731   46.240   29.847
O   34.022   45.706   31.578
C   32.831   45.372   32.401
C   31.635   50.768   34.017
C   31.418   52.290   34.108
C   31.186   53.067   35.213
C   30.869   52.472   36.587
C   31.001   54.544   35.016
C   32.219   55.398   35.542
C   32.227   56.930   35.075
C   32.766   57.968   36.140
C   34.343   57.826   35.942
C   32.259   59.440   35.904
C   31.332   60.001   36.985
C   30.437   61.165   36.559
C   29.003   60.957   37.046
C   28.558   59.654   36.274
C   28.135   62.169   36.658
C   26.888   62.277   37.545
C   25.749   63.046   36.831
C   25.961   64.612   36.931
C   24.657   65.109   37.719
C   26.124   65.391   35.716
H   33.181   52.834   27.589
H   34.481   51.383   21.508
H   35.607   45.421   23.288
H   33.056   48.971   30.392
H   34.091   51.819   29.907
H   31.344   50.947   29.006
H   32.023   52.444   29.593
H   31.808   51.036   30.740
H   35.381   50.778   31.224
H   35.847   49.347   30.764
H   35.362   49.623   33.215
H   34.877   48.102   32.448
H   34.388   54.856   26.436
H   32.620   54.588   26.190
H   33.495   55.679   25.164
H   34.533   56.031   23.491
H   32.834   55.656   22.965
H   34.084   56.070   21.788
H   35.585   49.186   20.551
H   34.746   46.645   21.260
H   33.065   47.658   20.265
H   33.375   49.253   19.606
H   32.666   49.087   21.193
H   37.069   47.312   20.008
H   36.831   45.871   20.960
H   37.992   47.897   22.613
H   38.830   47.847   21.021
H   38.634   46.371   22.042
H   36.613   43.917   24.846
H   35.144   43.092   25.518
H   36.539   43.210   26.521
H   35.845   46.784   30.399
H   33.189   44.923   33.328
H   32.385   44.497   31.930
H   32.104   46.161   32.592
H   31.804   50.376   35.019
H   30.803   50.214   33.581
H   31.527   52.790   33.146
H   30.012   52.941   37.070
H   31.693   52.649   37.279
H   30.762   51.403   36.405
H   30.151   54.937   35.573
H   30.741   54.797   33.988
H   33.177   54.955   35.271
H   32.176   55.304   36.627
H   31.201   57.254   34.900
H   32.661   56.943   34.075
H   32.480   57.660   37.146
H   34.818   57.157   36.659
H   34.812   58.796   36.105
H   34.595   57.511   34.929
H   31.728   59.476   34.953
H   33.039   60.184   35.743
H   31.936   60.459   37.768
H   30.831   59.180   37.497
H   30.320   61.199   35.476
H   30.942   62.073   36.888
H   29.117   60.920   38.129
H   28.253   58.888   36.986
H   29.323   59.111   35.720
H   27.760   59.824   35.551
H   27.913   62.047   35.598
H   28.743   63.052   36.853
H   27.297   62.878   38.358
H   26.512   61.291   37.817
H   24.862   62.640   37.317
H   25.814   62.669   35.811
H   26.822   64.805   37.571
H   23.872   64.353   37.686
H   24.182   66.003   37.316
H   24.823   65.418   38.751
H   25.210   65.782   35.269
H   26.696   64.735   35.059
H   26.674   66.316   35.887

