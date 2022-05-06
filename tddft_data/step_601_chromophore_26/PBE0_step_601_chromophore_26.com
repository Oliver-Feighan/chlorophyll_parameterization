%nproc=24
%mem=75GB
#p PBE1PBE/Def2SVP td=(nstates=5)

step_601_chromophore_26 TDDFT with PBE1PBE functional

0 1
Mg   -8.557   17.991   42.811
C   -5.115   17.521   42.637
C   -8.082   21.348   42.129
C   -11.831   18.262   42.333
C   -8.890   14.418   42.811
N   -6.810   19.322   42.418
C   -5.468   18.899   42.464
C   -4.456   20.035   42.231
C   -5.404   21.260   42.382
C   -6.880   20.638   42.203
C   -5.234   22.223   43.596
C   -3.652   19.952   40.931
C   -4.305   19.233   39.683
C   -3.919   19.864   38.335
O   -3.101   20.785   38.160
O   -4.666   19.212   37.349
N   -9.802   19.608   42.339
C   -9.426   20.920   42.128
C   -10.587   21.710   41.881
C   -11.748   20.869   41.916
C   -11.149   19.489   42.171
C   -10.543   23.204   41.535
C   -13.211   21.257   41.765
O   -13.495   22.462   41.780
C   -14.302   20.304   41.846
N   -10.087   16.462   42.372
C   -11.333   16.925   42.455
C   -12.341   15.745   42.841
C   -11.534   14.478   42.763
C   -10.086   15.179   42.669
C   -13.075   15.975   44.155
C   -11.812   13.454   41.625
C   -12.245   12.017   42.037
N   -7.294   16.190   42.870
C   -7.576   14.887   42.910
C   -6.348   14.179   42.917
C   -5.313   15.182   42.908
C   -5.964   16.356   42.816
C   -6.245   12.652   42.805
C   -3.873   15.396   42.947
O   -2.964   14.615   43.038
C   -3.748   16.947   42.779
C   -3.092   17.316   44.045
O   -1.924   17.766   44.047
O   -3.963   17.233   45.141
C   -3.351   17.442   46.439
C   -4.308   19.481   35.934
C   -5.575   19.399   35.058
C   -6.039   18.514   34.151
C   -5.561   17.080   33.999
C   -7.119   19.054   33.251
C   -8.583   19.003   33.931
C   -9.753   18.403   32.997
C   -11.063   19.342   33.164
C   -12.396   18.515   32.950
C   -10.970   20.472   32.034
C   -11.359   21.914   32.669
C   -10.070   22.822   32.645
C   -10.057   23.906   31.522
C   -9.386   25.201   32.053
C   -9.387   23.392   30.166
C   -10.049   23.909   28.832
C   -10.954   22.881   28.057
C   -10.993   23.264   26.554
C   -12.425   23.319   25.916
C   -9.995   22.447   25.740
H   -7.998   22.429   41.995
H   -12.915   18.322   42.453
H   -8.971   13.337   42.942
H   -3.778   19.819   43.056
H   -5.350   21.997   41.581
H   -4.370   21.812   44.118
H   -6.143   22.183   44.196
H   -5.018   23.239   43.264
H   -2.753   19.397   41.200
H   -3.244   20.935   40.694
H   -5.381   19.405   39.711
H   -4.216   18.146   39.686
H   -10.257   23.515   40.531
H   -9.933   23.772   42.238
H   -11.482   23.698   41.788
H   -14.406   19.842   42.827
H   -13.981   19.533   41.145
H   -15.172   20.856   41.489
H   -13.128   15.773   42.087
H   -11.600   13.943   43.710
H   -12.609   16.796   44.700
H   -13.171   15.037   44.701
H   -14.150   16.044   43.992
H   -10.962   13.342   40.952
H   -12.637   13.902   41.072
H   -11.754   11.178   41.544
H   -13.255   11.779   41.705
H   -12.182   11.900   43.119
H   -7.247   12.231   42.881
H   -5.554   12.279   43.561
H   -5.846   12.428   41.816
H   -2.993   17.195   42.032
H   -3.546   16.594   47.095
H   -3.693   18.395   46.840
H   -2.269   17.570   46.482
H   -3.672   18.643   35.649
H   -3.781   20.434   35.899
H   -5.903   20.438   35.057
H   -5.679   16.738   32.971
H   -6.233   16.484   34.616
H   -4.523   16.957   34.307
H   -7.110   18.633   32.246
H   -6.840   20.105   33.180
H   -8.886   20.034   34.113
H   -8.662   18.415   34.845
H   -10.061   17.437   33.399
H   -9.418   18.210   31.978
H   -11.186   19.786   34.152
H   -13.170   18.765   33.676
H   -12.229   17.468   33.205
H   -12.843   18.567   31.957
H   -11.596   20.395   31.145
H   -9.908   20.490   31.789
H   -11.718   21.834   33.695
H   -12.236   22.247   32.115
H   -9.118   22.307   32.521
H   -9.929   23.327   33.600
H   -11.097   24.128   31.283
H   -8.965   25.105   33.054
H   -10.097   26.020   31.948
H   -8.533   25.478   31.433
H   -9.168   22.324   30.183
H   -8.385   23.821   30.154
H   -9.237   24.162   28.152
H   -10.722   24.733   29.071
H   -11.907   22.920   28.585
H   -10.542   21.888   28.235
H   -10.646   24.297   26.580
H   -13.212   23.189   26.659
H   -12.710   22.630   25.120
H   -12.522   24.327   25.511
H   -9.256   23.050   25.214
H   -10.542   21.935   24.948
H   -9.475   21.716   26.359
