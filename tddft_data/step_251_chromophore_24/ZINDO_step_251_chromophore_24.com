%nproc=24
%mem=50gb
#p ZINDO=(nstates=5)

step_251_chromophore_24 ZINDO

0 1
Mg   -0.437   44.186   24.825
C   1.434   43.801   27.727
C   -3.182   43.128   26.591
C   -2.256   44.310   21.894
C   2.466   44.703   22.966
N   -0.821   43.638   26.941
C   0.041   43.653   27.972
C   -0.601   43.417   29.359
C   -2.064   42.961   28.940
C   -2.049   43.210   27.356
C   -2.282   41.470   29.382
C   -0.536   44.820   30.155
C   0.124   44.790   31.565
C   0.166   43.363   32.222
O   1.156   42.609   32.280
O   -1.082   42.941   32.698
N   -2.405   43.722   24.252
C   -3.412   43.377   25.177
C   -4.692   43.407   24.502
C   -4.450   43.691   23.152
C   -2.978   44.009   23.044
C   -5.959   42.859   25.174
C   -5.472   43.730   21.985
O   -5.197   43.996   20.800
C   -6.957   43.498   22.313
N   0.099   44.393   22.731
C   -0.908   44.467   21.721
C   -0.261   44.514   20.321
C   1.236   44.926   20.706
C   1.266   44.704   22.241
C   -0.474   43.183   19.619
C   1.529   46.446   20.357
C   0.657   47.664   20.816
N   1.536   44.172   25.194
C   2.632   44.380   24.351
C   3.831   44.316   25.122
C   3.427   44.139   26.433
C   2.023   43.949   26.438
C   5.257   44.369   24.613
C   3.897   44.019   27.756
O   5.030   43.965   28.115
C   2.621   43.676   28.631
C   2.820   42.268   29.120
O   2.222   41.307   28.683
O   3.773   42.301   30.059
C   4.377   41.012   30.292
C   -1.179   41.614   33.305
C   -2.334   41.356   34.167
C   -2.495   41.533   35.472
C   -1.598   42.275   36.443
C   -3.797   41.022   36.131
C   -5.039   41.756   35.826
C   -6.116   40.973   35.101
C   -7.548   41.135   35.706
C   -8.145   42.424   35.244
C   -8.444   39.897   35.437
C   -9.091   39.319   36.711
C   -10.575   39.638   36.797
C   -11.162   39.855   38.276
C   -11.859   38.612   38.831
C   -12.038   41.128   38.348
C   -11.369   42.448   37.761
C   -11.806   43.751   38.446
C   -13.271   44.090   38.019
C   -13.322   45.425   37.207
C   -14.277   44.156   39.158
H   -4.077   42.871   27.161
H   -2.791   44.110   20.964
H   3.386   44.852   22.396
H   -0.099   42.677   29.982
H   -2.869   43.574   29.346
H   -2.774   40.836   28.644
H   -2.909   41.649   30.255
H   -1.320   41.107   29.743
H   -1.542   45.239   30.147
H   0.042   45.503   29.532
H   -0.529   45.470   32.111
H   1.139   45.184   31.607
H   -6.353   42.079   24.522
H   -6.719   43.641   25.202
H   -5.721   42.495   26.173
H   -7.068   42.435   22.531
H   -7.491   43.661   21.377
H   -7.303   44.269   23.001
H   -0.720   45.376   19.836
H   1.942   44.368   20.092
H   -1.382   42.671   19.938
H   0.369   42.516   19.798
H   -0.613   43.329   18.548
H   1.659   46.516   19.277
H   2.469   46.628   20.878
H   -0.126   47.821   20.074
H   1.340   48.514   20.838
H   0.276   47.625   21.837
H   5.474   45.300   24.089
H   5.311   43.538   23.910
H   6.032   44.403   25.380
H   2.440   44.276   29.523
H   5.422   41.062   30.597
H   4.252   40.186   29.592
H   3.939   40.474   31.133
H   -0.307   41.188   33.802
H   -1.334   40.913   32.485
H   -2.976   40.592   33.729
H   -2.262   42.993   36.925
H   -0.726   42.747   35.989
H   -1.275   41.534   37.174
H   -3.616   40.855   37.193
H   -4.047   39.976   35.954
H   -4.827   42.606   35.177
H   -5.418   42.056   36.803
H   -5.869   39.933   34.885
H   -6.130   41.328   34.070
H   -7.334   41.221   36.771
H   -8.456   42.928   36.159
H   -9.022   42.275   34.614
H   -7.479   43.111   34.721
H   -7.815   39.101   35.039
H   -9.068   40.117   34.570
H   -8.525   39.801   37.508
H   -8.914   38.243   36.726
H   -11.096   38.837   36.272
H   -10.843   40.474   36.152
H   -10.303   40.177   38.864
H   -12.938   38.761   38.887
H   -11.532   38.597   39.870
H   -11.700   37.652   38.338
H   -12.486   41.294   39.327
H   -12.801   40.762   37.661
H   -11.605   42.525   36.700
H   -10.291   42.310   37.844
H   -11.136   44.583   38.231
H   -11.887   43.697   39.532
H   -13.687   43.243   37.472
H   -13.739   46.254   37.780
H   -14.120   45.259   36.483
H   -12.321   45.693   36.869
H   -15.010   44.951   39.022
H   -13.930   44.337   40.175
H   -14.881   43.251   39.091

