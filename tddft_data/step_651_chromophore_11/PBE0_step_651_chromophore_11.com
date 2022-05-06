%nproc=24
%mem=75GB
#p PBE1PBE/Def2SVP td=(nstates=5)

step_651_chromophore_11 TDDFT with PBE1PBE functional

0 1
Mg   52.249   23.793   45.265
C   49.423   25.695   44.137
C   50.625   21.067   43.983
C   55.185   22.016   45.173
C   53.926   26.784   45.702
N   50.221   23.511   44.223
C   49.227   24.323   43.878
C   47.994   23.709   43.308
C   48.382   22.189   43.657
C   49.835   22.224   43.974
C   47.588   21.546   44.775
C   47.720   24.105   41.805
C   48.974   24.125   40.912
C   48.683   24.427   39.514
O   48.653   25.544   39.012
O   48.182   23.279   38.838
N   52.812   21.752   44.831
C   51.948   20.854   44.336
C   52.670   19.574   44.286
C   53.992   19.791   44.783
C   54.071   21.216   44.995
C   52.024   18.250   43.848
C   55.113   18.740   45.005
O   54.923   17.560   44.632
C   56.507   19.124   45.606
N   54.248   24.304   45.403
C   55.266   23.373   45.409
C   56.717   24.061   45.583
C   56.249   25.571   45.671
C   54.741   25.571   45.533
C   57.672   23.617   46.702
C   56.976   26.477   44.594
C   57.719   27.607   45.249
N   51.838   25.851   45.035
C   52.532   26.906   45.376
C   51.677   28.109   45.402
C   50.474   27.696   44.926
C   50.576   26.329   44.718
C   52.072   29.486   45.781
C   49.085   28.057   44.717
O   48.514   29.117   44.939
C   48.447   26.807   44.025
C   48.271   27.314   42.670
O   49.136   27.646   41.893
O   46.928   27.411   42.345
C   46.733   27.968   41.038
C   47.884   23.466   37.421
C   48.771   22.467   36.665
C   48.433   21.245   36.206
C   47.063   20.597   36.664
C   49.261   20.375   35.322
C   50.551   19.771   35.890
C   51.861   20.083   35.055
C   52.640   18.793   34.498
C   54.186   18.912   34.820
C   52.323   18.731   32.984
C   50.953   18.019   32.859
C   51.013   16.500   32.878
C   50.498   15.789   31.624
C   50.098   14.357   31.936
C   51.581   15.782   30.582
C   51.000   16.393   29.295
C   52.160   16.549   28.161
C   51.920   15.551   26.870
C   52.911   14.312   27.038
C   52.156   16.338   25.619
H   50.017   20.202   43.708
H   56.147   21.500   45.196
H   54.487   27.671   46.006
H   47.117   23.971   43.899
H   48.245   21.552   42.783
H   47.139   20.620   44.415
H   46.748   22.200   45.009
H   48.131   21.335   45.696
H   47.166   25.040   41.900
H   47.091   23.370   41.302
H   49.430   23.157   41.119
H   49.673   24.858   41.316
H   51.966   18.252   42.759
H   50.990   18.145   44.174
H   52.588   17.372   44.162
H   57.016   19.806   44.924
H   57.012   18.159   45.585
H   56.205   19.583   46.547
H   57.108   23.937   44.573
H   56.434   25.929   46.684
H   57.282   23.718   47.715
H   58.694   23.970   46.566
H   57.769   22.532   46.677
H   56.337   26.815   43.777
H   57.749   25.886   44.103
H   58.708   27.336   45.621
H   57.161   28.031   46.084
H   58.017   28.335   44.495
H   52.284   30.204   44.988
H   52.961   29.349   46.397
H   51.257   29.861   46.400
H   47.453   26.569   44.404
H   45.650   28.090   41.007
H   47.039   27.280   40.250
H   47.234   28.936   41.021
H   47.916   24.477   37.016
H   46.854   23.116   37.343
H   49.612   22.925   36.144
H   47.113   20.673   37.750
H   47.028   19.541   36.395
H   46.109   21.038   36.376
H   49.536   20.934   34.427
H   48.603   19.659   34.831
H   50.264   18.720   35.912
H   50.816   20.074   36.902
H   52.508   20.722   35.656
H   51.602   20.833   34.309
H   52.227   17.952   35.055
H   54.744   18.359   34.064
H   54.365   18.294   35.700
H   54.657   19.892   34.897
H   53.022   18.036   32.521
H   52.306   19.751   32.602
H   50.618   18.342   31.874
H   50.236   18.390   33.592
H   50.416   16.166   33.727
H   51.976   16.146   33.247
H   49.599   16.333   31.334
H   49.013   14.451   31.885
H   50.476   14.160   32.940
H   50.435   13.641   31.186
H   52.137   14.852   30.467
H   52.421   16.447   30.785
H   50.515   17.357   29.444
H   50.347   15.601   28.928
H   53.128   16.348   28.620
H   52.264   17.582   27.829
H   50.902   15.162   26.887
H   52.928   14.152   28.116
H   53.931   14.494   26.700
H   52.502   13.510   26.423
H   51.504   17.168   25.346
H   52.083   15.668   24.762
H   53.160   16.761   25.588

