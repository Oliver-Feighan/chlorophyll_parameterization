%nproc=24
%mem=75GB
#p cam-b3lyp/Def2SVP td=(nstates=5)

step_301_chromophore_17 TDDFT with cam-b3lyp functional

0 1
Mg   29.411   59.430   41.775
C   26.283   58.083   40.766
C   30.959   56.526   40.450
C   32.464   60.824   42.089
C   27.798   62.231   42.798
N   28.676   57.607   40.591
C   27.406   57.272   40.375
C   27.415   55.998   39.619
C   28.828   55.413   39.753
C   29.538   56.523   40.451
C   28.984   54.006   40.265
C   26.813   56.078   38.135
C   26.113   54.833   37.704
C   26.646   54.262   36.383
O   27.397   53.295   36.377
O   26.220   54.983   35.271
N   31.442   58.796   41.231
C   31.928   57.514   40.786
C   33.270   57.454   40.635
C   33.700   58.773   40.966
C   32.528   59.536   41.481
C   33.973   56.225   40.081
C   35.075   59.221   40.876
O   35.952   58.412   40.567
C   35.564   60.632   41.140
N   30.067   61.296   42.301
C   31.380   61.643   42.444
C   31.527   63.083   42.950
C   29.989   63.540   43.109
C   29.222   62.324   42.625
C   32.298   63.276   44.299
C   29.630   64.853   42.254
C   29.100   65.957   43.202
N   27.440   59.991   41.912
C   26.931   61.209   42.411
C   25.453   61.033   42.519
C   25.196   59.848   41.828
C   26.434   59.309   41.399
C   24.544   62.131   42.901
C   24.098   58.978   41.313
O   22.893   59.001   41.422
C   24.831   57.784   40.622
C   24.383   56.492   41.179
O   24.343   56.263   42.349
O   24.008   55.623   40.209
C   23.484   54.422   40.896
C   26.685   54.534   33.998
C   26.539   55.853   33.157
C   26.504   55.816   31.807
C   26.702   54.530   30.935
C   26.202   57.056   31.008
C   27.442   57.790   30.318
C   27.443   57.679   28.776
C   28.590   56.752   28.226
C   28.255   55.968   26.977
C   29.975   57.546   28.100
C   31.263   56.630   28.264
C   32.272   57.029   29.312
C   32.450   56.129   30.547
C   31.313   56.509   31.620
C   33.870   56.225   31.124
C   34.604   54.901   30.916
C   36.159   55.013   31.161
C   36.956   54.903   29.750
C   37.701   53.494   29.673
C   37.749   56.173   29.358
H   31.471   55.644   40.059
H   33.400   61.288   42.405
H   27.432   63.187   43.178
H   26.748   55.405   40.245
H   29.242   55.588   38.760
H   29.758   54.142   41.020
H   29.314   53.437   39.396
H   28.012   53.626   40.581
H   27.633   56.185   37.425
H   26.169   56.939   37.954
H   25.064   55.021   37.477
H   26.121   54.058   38.470
H   34.033   56.344   39.000
H   33.613   55.214   40.275
H   34.928   56.048   40.577
H   34.722   61.286   40.916
H   36.446   60.831   40.531
H   35.753   60.599   42.214
H   32.108   63.555   42.157
H   29.898   63.723   44.180
H   31.957   64.079   44.952
H   33.295   63.442   43.889
H   32.275   62.328   44.837
H   28.861   64.606   41.522
H   30.496   65.151   41.663
H   29.519   66.945   43.014
H   29.156   65.592   44.228
H   28.022   66.078   43.100
H   23.627   61.645   43.233
H   24.431   62.728   41.996
H   24.856   62.700   43.777
H   24.542   57.755   39.572
H   24.365   53.876   41.231
H   22.931   53.805   40.188
H   22.750   54.622   41.677
H   26.041   53.745   33.609
H   27.726   54.212   34.022
H   26.297   56.815   33.610
H   25.932   53.832   31.263
H   27.688   54.129   31.171
H   26.657   54.685   29.857
H   25.745   57.741   31.722
H   25.415   56.821   30.292
H   28.201   57.096   30.679
H   27.597   58.809   30.672
H   27.567   58.673   28.346
H   26.539   57.103   28.580
H   28.721   55.977   28.981
H   28.651   56.568   26.158
H   27.177   55.832   26.902
H   28.799   55.025   26.917
H   30.018   58.327   28.859
H   30.080   58.129   27.185
H   31.701   56.588   27.267
H   30.912   55.628   28.511
H   31.999   58.031   29.643
H   33.272   57.105   28.885
H   32.117   55.138   30.239
H   30.696   57.367   31.352
H   31.818   56.667   32.573
H   30.675   55.633   31.736
H   33.870   56.459   32.188
H   34.389   57.062   30.657
H   34.491   54.496   29.910
H   34.230   54.213   31.674
H   36.371   54.248   31.908
H   36.435   55.922   31.696
H   36.325   54.818   28.866
H   38.688   53.524   29.212
H   37.036   52.833   29.116
H   37.777   52.906   30.587
H   37.195   56.960   28.847
H   38.590   55.833   28.753
H   38.154   56.641   30.256

