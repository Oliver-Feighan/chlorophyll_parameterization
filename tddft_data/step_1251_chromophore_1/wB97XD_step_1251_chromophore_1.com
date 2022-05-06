%nproc=24
%mem=75GB
#p wB97XD/Def2SVP td=(nstates=5)

step_1251_chromophore_1 TDDFT with wB97XD functional

0 1
Mg   -1.018   17.271   26.855
C   -1.464   15.185   29.792
C   -2.078   19.951   28.666
C   -0.819   19.038   24.044
C   -0.489   14.329   24.963
N   -1.593   17.506   28.937
C   -1.656   16.603   29.999
C   -1.767   17.339   31.377
C   -2.279   18.741   30.935
C   -2.015   18.732   29.449
C   -3.783   19.018   31.156
C   -0.662   17.400   32.394
C   -1.017   18.090   33.747
C   -1.062   17.215   34.965
O   -1.930   16.416   35.315
O   0.092   17.396   35.673
N   -1.284   19.281   26.460
C   -1.723   20.234   27.317
C   -1.703   21.454   26.705
C   -1.267   21.297   25.362
C   -1.115   19.806   25.231
C   -2.179   22.723   27.451
C   -1.052   22.336   24.257
O   -0.938   21.965   23.071
C   -0.953   23.846   24.486
N   -0.775   16.753   24.772
C   -0.899   17.661   23.773
C   -0.786   17.079   22.412
C   -0.206   15.761   22.658
C   -0.500   15.556   24.241
C   -2.100   17.189   21.584
C   1.272   15.402   22.431
C   2.485   16.304   22.739
N   -0.921   15.203   27.223
C   -0.813   14.174   26.382
C   -0.823   12.873   27.076
C   -1.043   13.201   28.375
C   -1.160   14.644   28.453
C   -0.983   11.473   26.472
C   -1.307   12.689   29.750
O   -1.536   11.571   30.144
C   -1.628   13.961   30.671
C   -0.737   13.792   31.807
O   0.424   14.295   31.846
O   -1.360   12.975   32.776
C   -0.487   12.969   33.988
C   0.235   16.652   36.927
C   0.119   17.607   38.081
C   0.102   17.417   39.382
C   0.107   16.004   39.918
C   0.074   18.548   40.478
C   1.272   19.524   40.771
C   1.008   21.043   40.959
C   2.105   21.949   40.264
C   2.054   22.049   38.804
C   3.463   21.630   40.974
C   3.971   22.791   41.891
C   4.776   22.291   43.068
C   4.109   22.514   44.453
C   5.017   21.962   45.535
C   2.763   21.580   44.600
C   1.666   22.595   44.771
C   0.286   21.899   44.846
C   -0.565   22.458   45.937
C   0.056   22.079   47.271
C   -2.089   22.111   45.852
H   -2.295   20.859   29.232
H   -0.741   19.590   23.105
H   -0.207   13.449   24.381
H   -2.646   16.801   31.732
H   -1.669   19.494   31.433
H   -4.330   19.110   30.218
H   -3.947   20.020   31.551
H   -4.312   18.238   31.703
H   0.244   17.843   31.980
H   -0.243   16.429   32.656
H   -2.055   18.417   33.694
H   -0.344   18.902   34.024
H   -1.435   23.203   28.087
H   -2.982   22.355   28.091
H   -2.638   23.358   26.693
H   -0.629   24.317   23.558
H   -0.135   24.025   25.184
H   -1.939   24.074   24.892
H   -0.020   17.626   21.862
H   -0.833   15.080   22.082
H   -2.872   17.798   22.055
H   -2.472   16.195   21.335
H   -1.819   17.671   20.648
H   1.353   15.175   21.368
H   1.441   14.494   23.010
H   2.875   16.865   21.890
H   3.183   15.664   23.278
H   2.066   17.070   23.391
H   -0.000   11.091   26.197
H   -1.606   11.565   25.582
H   -1.354   10.782   27.229
H   -2.657   13.925   31.029
H   0.409   12.347   33.959
H   -1.184   12.570   34.724
H   -0.279   13.999   34.277
H   1.238   16.235   37.013
H   -0.493   15.886   37.192
H   -0.018   18.646   37.782
H   -0.920   15.707   40.130
H   0.565   15.811   40.888
H   0.550   15.240   39.280
H   -0.272   18.136   41.426
H   -0.779   19.157   40.178
H   1.868   19.253   39.898
H   1.903   19.112   41.558
H   1.180   21.206   42.023
H   -0.021   21.288   40.694
H   1.732   22.889   40.669
H   1.400   21.405   38.217
H   3.027   21.681   38.480
H   1.980   23.128   38.666
H   4.239   21.457   40.228
H   3.325   20.718   41.556
H   3.114   23.357   42.257
H   4.614   23.476   41.339
H   5.755   22.765   43.125
H   5.121   21.261   42.970
H   4.122   23.604   44.469
H   5.068   20.875   45.479
H   4.728   22.391   46.495
H   6.018   22.351   45.347
H   2.838   21.112   45.582
H   2.447   20.943   43.774
H   1.738   23.258   43.909
H   1.846   23.238   45.632
H   0.487   20.832   44.946
H   -0.228   22.022   43.893
H   -0.473   23.544   45.956
H   0.485   21.080   47.197
H   -0.688   22.091   48.068
H   0.733   22.870   47.596
H   -2.434   21.706   46.804
H   -2.282   21.357   45.089
H   -2.450   23.073   45.488

