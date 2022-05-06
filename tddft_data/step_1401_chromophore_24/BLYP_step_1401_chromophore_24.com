%nproc=24
%mem=75GB
#p blyp/Def2SVP td=(nstates=5)

step_1401_chromophore_24 TDDFT with blyp functional

0 1
Mg   0.177   43.474   25.517
C   2.374   43.055   28.226
C   -2.514   42.911   27.703
C   -2.016   43.361   22.988
C   2.819   43.929   23.403
N   0.035   43.093   27.644
C   0.985   43.100   28.569
C   0.437   42.977   30.007
C   -0.995   42.549   29.767
C   -1.191   42.940   28.309
C   -1.318   41.028   30.013
C   0.446   44.295   30.875
C   -0.141   44.228   32.320
C   -0.145   42.749   32.933
O   0.850   42.118   33.292
O   -1.464   42.279   33.062
N   -1.983   43.078   25.357
C   -2.884   42.876   26.379
C   -4.255   42.786   25.863
C   -4.129   42.870   24.446
C   -2.693   43.096   24.184
C   -5.432   42.479   26.721
C   -5.221   42.756   23.438
O   -5.004   43.006   22.266
C   -6.635   42.416   23.860
N   0.375   43.545   23.503
C   -0.700   43.522   22.664
C   -0.125   43.556   21.264
C   1.313   43.975   21.382
C   1.538   43.735   22.839
C   -0.364   42.121   20.648
C   1.631   45.420   20.939
C   1.018   46.588   21.717
N   2.213   43.448   25.729
C   3.204   43.702   24.782
C   4.557   43.462   25.403
C   4.218   43.294   26.763
C   2.793   43.361   26.902
C   5.899   43.314   24.792
C   4.764   42.993   28.028
O   5.941   42.713   28.377
C   3.661   43.043   29.052
C   3.656   41.869   29.893
O   3.596   40.712   29.519
O   3.933   42.345   31.157
C   4.237   41.274   32.099
C   -1.541   40.988   33.886
C   -2.774   41.147   34.769
C   -2.838   41.882   35.922
C   -1.700   42.868   36.270
C   -4.181   42.163   36.571
C   -4.946   43.400   36.037
C   -6.282   43.012   35.370
C   -7.521   43.095   36.324
C   -8.192   44.481   36.192
C   -8.589   42.023   36.092
C   -8.839   41.110   37.270
C   -9.815   41.794   38.228
C   -11.297   41.147   38.261
C   -12.338   42.290   38.213
C   -11.452   40.302   39.535
C   -11.962   38.873   39.295
C   -13.320   38.566   40.025
C   -14.567   38.754   39.132
C   -15.863   38.775   39.988
C   -14.652   37.552   38.069
H   -3.267   42.749   28.477
H   -2.618   43.234   22.086
H   3.649   44.132   22.723
H   0.970   42.202   30.558
H   -1.716   43.203   30.258
H   -1.536   40.505   29.082
H   -2.147   40.897   30.708
H   -0.383   40.597   30.372
H   -0.031   45.107   30.326
H   1.444   44.723   30.962
H   -1.098   44.744   32.397
H   0.425   44.855   33.009
H   -5.924   43.350   27.154
H   -4.972   41.907   27.526
H   -6.036   41.671   26.307
H   -7.044   43.019   24.671
H   -6.606   41.382   24.205
H   -7.300   42.522   23.003
H   -0.735   44.271   20.713
H   1.998   43.280   20.897
H   0.575   41.572   20.577
H   -0.820   42.241   19.665
H   -0.978   41.591   21.376
H   1.275   45.397   19.909
H   2.714   45.533   20.988
H   1.832   47.309   21.783
H   0.509   46.397   22.662
H   0.353   47.128   21.044
H   5.808   42.811   23.829
H   6.566   42.818   25.497
H   6.282   44.318   24.610
H   3.701   44.003   29.566
H   4.642   41.730   33.002
H   4.946   40.612   31.603
H   3.310   40.742   32.315
H   -0.698   40.678   34.504
H   -1.785   40.136   33.252
H   -3.711   40.733   34.395
H   -1.008   42.435   36.992
H   -2.269   43.600   36.843
H   -1.220   43.362   35.425
H   -4.039   42.118   37.650
H   -4.747   41.247   36.401
H   -4.396   44.053   35.360
H   -5.113   44.063   36.886
H   -6.322   41.965   35.072
H   -6.449   43.633   34.490
H   -7.155   43.099   37.350
H   -8.595   44.862   37.131
H   -9.060   44.471   35.533
H   -7.568   45.244   35.727
H   -8.447   41.479   35.158
H   -9.505   42.519   35.771
H   -7.941   40.907   37.853
H   -9.193   40.099   37.065
H   -9.865   42.873   38.081
H   -9.255   41.654   39.152
H   -11.334   40.506   37.380
H   -13.283   42.140   38.735
H   -12.337   42.722   37.212
H   -11.904   43.143   38.734
H   -12.164   40.774   40.212
H   -10.491   40.197   40.039
H   -11.181   38.213   39.673
H   -12.057   38.564   38.255
H   -13.356   39.203   40.909
H   -13.316   37.517   40.321
H   -14.570   39.692   38.576
H   -16.314   39.763   40.082
H   -15.699   38.526   41.036
H   -16.616   38.183   39.467
H   -13.725   36.991   38.191
H   -14.673   38.014   37.082
H   -15.480   36.857   38.208

