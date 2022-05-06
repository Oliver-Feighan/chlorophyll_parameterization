%nproc=24
%mem=75GB
#p wB97XD/Def2SVP td=(nstates=5)

step_601_chromophore_5 TDDFT with wB97XD functional

0 1
Mg   24.460   -7.511   46.114
C   26.902   -5.127   45.059
C   22.114   -5.991   44.136
C   22.414   -10.023   46.780
C   27.133   -9.261   47.631
N   24.542   -5.794   44.816
C   25.611   -4.918   44.601
C   25.229   -3.790   43.591
C   23.629   -3.996   43.412
C   23.428   -5.355   44.272
C   22.761   -2.826   43.914
C   25.928   -3.863   42.309
C   26.481   -2.530   41.765
C   25.520   -1.895   40.739
O   24.422   -1.390   40.926
O   25.923   -2.227   39.482
N   22.380   -7.863   45.686
C   21.635   -7.170   44.820
C   20.339   -7.775   44.787
C   20.342   -8.953   45.592
C   21.726   -8.987   46.111
C   19.148   -7.063   44.163
C   19.166   -9.879   45.835
O   18.200   -9.712   45.074
C   19.223   -10.989   46.778
N   24.811   -9.316   46.978
C   23.754   -10.157   47.127
C   24.141   -11.471   47.892
C   25.642   -11.247   48.239
C   25.877   -9.900   47.526
C   23.260   -11.940   49.137
C   26.740   -12.310   48.086
C   27.493   -12.857   49.316
N   26.565   -7.340   46.281
C   27.537   -8.109   46.971
C   28.854   -7.489   46.880
C   28.676   -6.328   46.114
C   27.269   -6.289   45.773
C   30.015   -7.930   47.677
C   29.320   -5.049   45.639
O   30.431   -4.643   45.875
C   28.219   -4.341   44.798
C   28.231   -2.966   45.053
O   28.915   -2.198   44.354
O   27.490   -2.575   46.138
C   27.507   -1.193   46.451
C   25.074   -1.798   38.314
C   24.050   -2.778   38.007
C   24.142   -3.911   37.234
C   25.473   -4.610   36.968
C   22.979   -4.797   36.946
C   22.623   -5.084   35.522
C   22.323   -6.619   35.296
C   22.533   -7.057   33.803
C   21.425   -8.061   33.412
C   23.988   -7.602   33.531
C   24.352   -7.504   32.023
C   25.385   -6.377   31.718
C   24.815   -5.329   30.627
C   24.181   -4.224   31.415
C   25.906   -4.864   29.665
C   25.950   -5.456   28.225
C   25.174   -4.553   27.212
C   24.310   -5.402   26.232
C   25.151   -6.080   25.087
C   23.249   -4.536   25.598
H   21.357   -5.514   43.510
H   21.721   -10.801   47.105
H   28.050   -9.683   48.048
H   25.436   -2.802   44.002
H   23.308   -4.140   42.380
H   23.321   -2.046   44.429
H   21.933   -3.118   44.561
H   22.198   -2.196   43.225
H   25.173   -4.211   41.605
H   26.769   -4.557   42.313
H   27.480   -2.791   41.416
H   26.711   -1.805   42.545
H   19.306   -6.005   43.958
H   18.219   -7.301   44.682
H   19.006   -7.448   43.154
H   19.495   -10.519   47.723
H   20.030   -11.631   46.426
H   18.312   -11.581   46.863
H   23.902   -12.251   47.169
H   25.691   -11.002   49.300
H   22.923   -12.948   48.897
H   22.481   -11.207   49.347
H   23.909   -11.991   50.010
H   27.395   -11.985   47.278
H   26.276   -13.155   47.577
H   27.039   -13.848   49.337
H   27.140   -12.217   50.124
H   28.571   -12.842   49.154
H   29.754   -8.472   48.587
H   30.752   -7.148   47.859
H   30.583   -8.683   47.131
H   28.445   -4.585   43.760
H   27.070   -1.187   47.449
H   27.049   -0.594   45.664
H   28.504   -0.788   46.627
H   25.694   -1.686   37.424
H   24.593   -0.843   38.528
H   23.064   -2.462   38.348
H   25.295   -5.073   35.997
H   25.559   -5.439   37.670
H   26.315   -3.922   36.890
H   22.098   -4.356   37.413
H   23.113   -5.697   37.546
H   23.410   -4.781   34.831
H   21.741   -4.457   35.392
H   21.295   -6.748   35.635
H   22.969   -7.106   36.026
H   22.356   -6.189   33.168
H   20.746   -7.608   32.689
H   20.782   -8.313   34.254
H   21.898   -8.975   33.053
H   24.097   -8.657   33.780
H   24.661   -7.068   34.202
H   23.469   -7.443   31.386
H   24.797   -8.462   31.753
H   26.340   -6.724   31.324
H   25.615   -5.861   32.651
H   23.996   -5.801   30.084
H   24.581   -3.241   31.165
H   24.285   -4.372   32.490
H   23.122   -4.220   31.158
H   26.901   -4.994   30.091
H   25.783   -3.811   29.413
H   25.584   -6.483   28.230
H   26.998   -5.585   27.957
H   25.926   -4.006   26.643
H   24.516   -3.920   27.806
H   23.859   -6.159   26.873
H   25.745   -5.368   24.514
H   24.462   -6.712   24.527
H   25.844   -6.829   25.471
H   23.539   -4.234   24.591
H   22.881   -3.704   26.198
H   22.305   -5.081   25.585

