%nproc=24
%mem=75GB
#p PBE1PBE/Def2SVP td=(nstates=5)

step_151_chromophore_22 TDDFT with PBE1PBE functional

0 1
Mg   8.724   48.259   24.786
C   6.503   48.523   27.474
C   11.146   49.438   26.611
C   10.782   47.843   22.126
C   5.900   47.507   22.711
N   8.794   48.886   26.863
C   7.798   48.932   27.735
C   8.360   49.614   28.940
C   9.618   50.408   28.383
C   9.878   49.542   27.230
C   9.423   51.886   27.925
C   8.673   48.543   30.068
C   8.606   49.166   31.433
C   7.755   48.525   32.492
O   6.542   48.686   32.579
O   8.468   47.602   33.312
N   10.746   48.558   24.448
C   11.604   49.076   25.301
C   12.852   49.224   24.731
C   12.822   48.651   23.408
C   11.417   48.309   23.271
C   14.092   49.815   25.466
C   13.880   48.607   22.347
O   13.583   48.281   21.195
C   15.324   48.869   22.697
N   8.431   47.778   22.667
C   9.469   47.674   21.755
C   8.922   47.127   20.426
C   7.420   47.118   20.603
C   7.232   47.463   22.112
C   9.367   47.974   19.217
C   6.607   45.831   20.169
C   5.164   46.046   19.672
N   6.659   48.052   25.004
C   5.667   47.832   24.098
C   4.389   47.834   24.801
C   4.692   47.934   26.130
C   6.062   48.191   26.248
C   2.955   47.746   24.216
C   4.063   48.017   27.404
O   2.888   47.867   27.767
C   5.235   48.360   28.406
C   5.369   47.328   29.460
O   6.005   46.266   29.356
O   4.539   47.718   30.508
C   4.423   46.589   31.478
C   7.581   47.018   34.326
C   8.506   46.680   35.473
C   8.218   46.252   36.699
C   6.780   45.943   37.192
C   9.331   45.916   37.660
C   9.121   46.679   38.985
C   10.258   47.752   39.210
C   10.967   47.472   40.607
C   11.388   48.751   41.327
C   12.189   46.516   40.341
C   12.409   45.467   41.453
C   13.793   44.888   41.347
C   14.090   43.941   42.596
C   15.074   42.816   42.028
C   14.648   44.726   43.843
C   13.942   44.365   45.187
C   13.680   45.602   46.116
C   12.270   46.336   45.862
C   12.422   47.730   45.459
C   11.264   46.256   46.970
H   11.840   50.130   27.092
H   11.428   47.410   21.360
H   5.001   47.084   22.257
H   7.560   50.275   29.274
H   10.387   50.302   29.147
H   8.414   52.263   28.094
H   9.821   52.112   26.936
H   10.087   52.551   28.478
H   9.620   48.108   29.746
H   7.914   47.765   29.987
H   8.281   50.198   31.296
H   9.586   49.174   31.910
H   13.771   50.167   26.447
H   14.481   50.696   24.955
H   14.763   48.988   25.695
H   15.659   48.295   23.561
H   15.309   49.951   22.827
H   15.956   48.575   21.859
H   9.351   46.129   20.339
H   6.968   48.032   20.218
H   9.380   49.023   19.514
H   8.508   48.089   18.556
H   10.306   47.620   18.793
H   6.563   45.030   20.907
H   7.074   45.536   19.230
H   4.965   45.352   18.856
H   4.994   47.038   19.253
H   4.422   45.997   20.469
H   3.061   47.923   23.145
H   2.207   48.400   24.664
H   2.560   46.734   24.307
H   5.145   49.343   28.868
H   3.621   45.872   31.302
H   4.028   47.043   32.387
H   5.369   46.105   31.720
H   7.213   46.028   34.058
H   6.818   47.688   34.723
H   9.570   46.768   35.253
H   6.029   45.757   36.425
H   6.371   46.770   37.774
H   6.843   45.016   37.762
H   10.316   46.186   37.282
H   9.337   44.868   37.962
H   9.125   46.025   39.857
H   8.188   47.226   38.856
H   9.816   48.748   39.231
H   10.840   47.814   38.290
H   10.226   47.018   41.265
H   10.626   49.086   42.030
H   11.429   49.551   40.588
H   12.359   48.745   41.824
H   13.058   47.103   40.043
H   12.004   45.897   39.463
H   11.690   44.690   41.193
H   12.223   45.906   42.433
H   14.514   45.704   41.306
H   13.908   44.349   40.406
H   13.199   43.372   42.861
H   14.573   41.857   42.157
H   15.960   42.766   42.661
H   15.431   42.841   40.998
H   14.524   45.795   43.671
H   15.697   44.527   44.062
H   14.517   43.661   45.787
H   12.998   43.858   44.987
H   14.573   46.206   45.956
H   13.711   45.392   47.186
H   11.804   45.738   45.079
H   13.357   47.906   44.927
H   12.465   48.378   46.334
H   11.652   47.976   44.727
H   11.484   46.751   47.916
H   11.042   45.227   47.253
H   10.375   46.799   46.649

