%nproc=24
%mem=75GB
#p PBE1PBE/Def2SVP td=(nstates=5)

step_701_chromophore_1 TDDFT with PBE1PBE functional

0 1
Mg   -1.524   17.532   27.015
C   -2.294   15.372   29.605
C   -2.526   20.154   29.115
C   -1.219   19.687   24.412
C   -1.495   14.882   24.752
N   -2.398   17.648   29.079
C   -2.478   16.642   30.026
C   -2.834   17.208   31.381
C   -3.429   18.604   30.990
C   -2.645   18.879   29.736
C   -4.976   18.545   30.720
C   -1.602   17.378   32.313
C   -1.867   17.175   33.751
C   -0.623   16.807   34.562
O   0.502   16.590   34.100
O   -1.000   16.726   35.914
N   -1.702   19.604   26.844
C   -2.042   20.532   27.849
C   -1.876   21.845   27.331
C   -1.393   21.753   26.009
C   -1.441   20.294   25.693
C   -2.289   22.983   28.171
C   -1.014   22.757   25.041
O   -0.671   22.547   23.906
C   -1.064   24.214   25.481
N   -1.492   17.390   24.910
C   -1.396   18.412   24.037
C   -1.226   17.899   22.528
C   -1.076   16.368   22.725
C   -1.336   16.186   24.239
C   -2.493   18.307   21.647
C   0.286   15.712   22.334
C   1.532   16.296   22.992
N   -1.830   15.528   27.122
C   -1.764   14.546   26.090
C   -1.895   13.210   26.690
C   -2.049   13.523   28.064
C   -2.031   14.876   28.265
C   -1.847   11.917   26.034
C   -2.218   12.947   29.372
O   -2.265   11.789   29.786
C   -2.428   14.086   30.381
C   -1.500   13.980   31.553
O   -0.323   14.224   31.570
O   -2.201   13.702   32.647
C   -1.492   13.542   33.961
C   0.020   16.657   36.910
C   -0.493   17.637   37.930
C   -0.715   17.458   39.272
C   -0.367   16.305   40.198
C   -1.309   18.687   39.924
C   -0.171   19.454   40.635
C   -0.370   21.025   40.561
C   1.018   21.789   40.608
C   1.630   21.871   39.208
C   2.055   21.257   41.538
C   2.917   22.358   42.067
C   3.966   21.841   43.002
C   4.381   22.974   44.045
C   5.945   23.153   43.956
C   3.894   22.480   45.469
C   2.640   23.210   45.926
C   1.619   22.375   46.740
C   0.116   22.437   46.324
C   -0.829   21.861   47.372
C   -0.026   21.734   44.951
H   -2.905   20.900   29.817
H   -1.084   20.381   23.580
H   -1.509   14.029   24.070
H   -3.534   16.529   31.868
H   -3.216   19.364   31.742
H   -5.200   18.830   29.692
H   -5.407   19.273   31.407
H   -5.299   17.522   30.910
H   -1.250   18.409   32.282
H   -0.824   16.725   31.918
H   -2.501   16.290   33.798
H   -2.405   18.067   34.072
H   -1.443   23.583   28.505
H   -2.702   22.812   29.165
H   -3.068   23.605   27.729
H   -0.256   24.329   26.203
H   -2.081   24.329   25.854
H   -0.862   24.884   24.645
H   -0.365   18.307   21.997
H   -1.798   15.961   22.017
H   -3.288   18.659   22.304
H   -2.688   17.581   20.857
H   -2.303   19.233   21.104
H   0.404   15.831   21.257
H   0.165   14.643   22.507
H   1.280   16.952   23.825
H   2.118   16.859   22.267
H   2.144   15.440   23.277
H   -1.253   11.360   26.760
H   -1.418   11.831   25.036
H   -2.834   11.458   26.077
H   -3.438   14.084   30.790
H   -2.014   14.023   34.788
H   -0.476   13.935   33.948
H   -1.458   12.477   34.194
H   1.031   16.925   36.604
H   -0.012   15.695   37.422
H   -0.740   18.623   37.537
H   0.495   16.617   40.788
H   -0.179   15.375   39.663
H   -1.249   16.184   40.828
H   -2.009   18.433   40.720
H   -1.904   19.290   39.238
H   0.800   19.152   40.243
H   -0.125   19.241   41.703
H   -0.923   21.315   41.455
H   -1.027   21.198   39.709
H   0.700   22.777   40.940
H   1.005   21.210   38.608
H   2.617   21.411   39.170
H   1.720   22.865   38.770
H   2.839   20.556   41.251
H   1.452   20.887   42.367
H   2.352   23.194   42.480
H   3.564   22.795   41.306
H   4.792   21.480   42.388
H   3.727   20.973   43.616
H   3.947   23.956   43.858
H   6.474   23.036   44.902
H   6.092   24.195   43.672
H   6.475   22.577   43.198
H   4.725   22.852   46.068
H   3.739   21.415   45.641
H   2.081   23.782   45.186
H   3.061   23.986   46.565
H   1.706   22.898   47.693
H   1.844   21.317   46.871
H   -0.162   23.489   46.262
H   -1.667   22.522   47.594
H   -0.354   21.782   48.350
H   -1.339   20.958   47.037
H   -1.012   21.300   44.790
H   0.729   20.982   44.722
H   -0.009   22.423   44.106

