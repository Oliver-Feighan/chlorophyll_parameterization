%nproc=24
%mem=75GB
#p PBE1PBE/Def2SVP td=(nstates=5)

step_251_chromophore_13 TDDFT with PBE1PBE functional

0 1
Mg   46.817   25.044   28.063
C   47.435   27.224   30.704
C   46.222   22.545   30.243
C   46.683   22.915   25.441
C   47.912   27.510   25.916
N   46.855   24.932   30.276
C   46.927   25.999   31.135
C   46.692   25.556   32.529
C   46.355   24.062   32.460
C   46.450   23.768   30.901
C   47.140   23.089   33.326
C   45.548   26.464   33.259
C   45.965   27.280   34.505
C   45.045   27.400   35.695
O   43.784   27.238   35.608
O   45.716   27.572   36.878
N   46.380   23.046   27.855
C   46.172   22.149   28.892
C   45.972   20.849   28.363
C   46.106   20.923   26.950
C   46.382   22.340   26.692
C   45.547   19.695   29.353
C   46.048   19.732   25.995
O   46.242   19.783   24.771
C   45.659   18.414   26.504
N   47.462   25.070   25.979
C   47.201   24.135   25.089
C   47.753   24.524   23.699
C   47.640   26.091   23.862
C   47.646   26.260   25.378
C   49.211   24.116   23.270
C   46.370   26.803   23.190
C   45.026   26.386   23.628
N   47.607   27.038   28.187
C   48.008   27.951   27.192
C   48.427   29.169   27.809
C   48.294   28.934   29.163
C   47.685   27.648   29.336
C   49.119   30.268   27.164
C   48.307   29.557   30.402
O   48.645   30.694   30.747
C   47.854   28.483   31.493
C   49.084   28.177   32.282
O   50.165   27.734   31.955
O   48.788   28.534   33.533
C   49.860   28.299   34.555
C   44.875   27.588   38.067
C   44.835   26.212   38.647
C   44.167   25.775   39.768
C   43.255   26.699   40.533
C   44.371   24.391   40.339
C   43.093   23.748   40.810
C   42.966   23.612   42.334
C   41.528   23.320   42.867
C   40.958   22.030   42.262
C   40.525   24.527   42.742
C   39.516   24.665   43.891
C   40.079   25.230   45.187
C   40.111   24.142   46.249
C   38.544   23.911   46.573
C   40.836   24.562   47.567
C   42.070   23.696   47.932
C   43.453   24.400   48.202
C   44.667   23.570   47.696
C   45.891   23.430   48.677
C   45.100   24.218   46.328
H   45.835   21.892   31.028
H   46.476   22.230   24.617
H   48.193   28.302   25.219
H   47.645   25.712   33.033
H   45.304   23.926   32.714
H   47.873   22.821   32.566
H   46.566   22.268   33.756
H   47.692   23.458   34.190
H   44.704   25.823   33.512
H   45.170   27.275   32.636
H   46.056   28.313   34.168
H   46.967   27.023   34.846
H   46.224   18.853   29.210
H   44.591   19.468   28.881
H   45.601   19.885   30.425
H   46.323   18.092   27.305
H   45.949   17.706   25.728
H   44.600   18.298   26.736
H   46.986   24.305   22.956
H   48.530   26.463   23.355
H   49.971   24.189   24.047
H   49.403   24.792   22.436
H   49.235   23.084   22.919
H   46.327   26.676   22.108
H   46.289   27.843   23.506
H   44.885   26.945   24.552
H   44.844   25.338   23.868
H   44.172   26.645   23.003
H   50.158   29.937   27.143
H   49.062   31.226   27.680
H   48.710   30.524   26.186
H   47.000   28.805   32.089
H   49.507   27.436   35.120
H   49.990   29.194   35.163
H   50.848   28.017   34.190
H   43.834   27.893   37.963
H   45.304   28.139   38.904
H   45.519   25.527   38.145
H   43.411   26.606   41.608
H   42.201   26.477   40.362
H   43.441   27.734   40.247
H   45.125   24.571   41.105
H   44.872   23.717   39.643
H   43.188   22.751   40.379
H   42.145   24.051   40.365
H   43.407   24.481   42.822
H   43.560   22.779   42.711
H   41.774   23.182   43.920
H   40.832   21.365   43.116
H   41.698   21.606   41.583
H   40.002   22.272   41.798
H   39.998   24.368   41.801
H   41.032   25.464   42.513
H   39.452   23.577   43.890
H   38.590   25.045   43.460
H   39.366   26.023   45.410
H   41.050   25.681   44.980
H   40.544   23.209   45.888
H   38.418   22.923   46.130
H   37.780   24.588   46.192
H   38.330   23.843   47.640
H   40.127   24.723   48.379
H   41.209   25.535   47.248
H   42.232   22.886   47.220
H   41.795   23.201   48.863
H   43.543   24.493   49.284
H   43.496   25.415   47.807
H   44.457   22.522   47.480
H   46.502   22.698   48.150
H   45.486   22.977   49.582
H   46.504   24.297   48.923
H   44.288   24.780   45.868
H   45.502   23.518   45.596
H   45.940   24.867   46.578

