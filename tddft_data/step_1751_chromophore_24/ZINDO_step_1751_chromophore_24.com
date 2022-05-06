%nproc=24
%mem=50gb
#p ZINDO=(nstates=5)

step_1751_chromophore_24 ZINDO

0 1
Mg   -0.496   44.002   25.203
C   2.051   43.353   27.531
C   -2.854   43.228   27.384
C   -2.669   43.746   22.614
C   2.113   44.184   22.672
N   -0.363   43.340   27.181
C   0.761   43.273   28.021
C   0.346   43.010   29.510
C   -1.270   42.883   29.429
C   -1.544   43.144   27.913
C   -1.926   41.604   29.955
C   0.811   44.127   30.500
C   0.821   43.766   31.942
C   0.221   42.431   32.479
O   0.815   41.367   32.339
O   -0.999   42.527   33.146
N   -2.493   43.891   25.062
C   -3.274   43.529   26.105
C   -4.667   43.600   25.642
C   -4.653   43.851   24.209
C   -3.226   43.812   23.901
C   -5.885   43.373   26.608
C   -5.845   43.897   23.285
O   -5.641   44.070   22.108
C   -7.110   43.649   23.771
N   -0.294   44.331   22.905
C   -1.335   43.866   22.182
C   -0.851   43.634   20.670
C   0.589   44.163   20.677
C   0.852   44.307   22.159
C   -1.046   42.153   20.216
C   1.019   45.452   19.920
C   0.610   46.802   20.657
N   1.683   43.976   25.077
C   2.556   43.964   24.027
C   3.917   43.884   24.550
C   3.723   43.628   25.902
C   2.390   43.636   26.185
C   5.181   43.848   23.683
C   4.456   43.340   27.113
O   5.615   43.085   27.243
C   3.434   43.181   28.230
C   3.678   41.810   28.835
O   3.575   40.769   28.244
O   4.142   41.890   30.163
C   4.153   40.594   30.877
C   -1.274   41.269   33.910
C   -2.621   41.298   34.497
C   -3.037   41.326   35.801
C   -2.140   41.407   36.971
C   -4.573   41.258   36.066
C   -5.507   42.475   35.827
C   -6.966   42.090   35.355
C   -8.112   42.567   36.249
C   -8.412   44.129   36.001
C   -9.396   41.756   35.974
C   -9.886   41.109   37.343
C   -10.910   41.969   38.003
C   -12.367   41.657   37.363
C   -13.036   42.886   36.685
C   -13.333   41.038   38.461
C   -14.403   40.090   37.906
C   -14.187   38.690   38.563
C   -13.251   37.797   37.715
C   -14.171   36.784   37.053
C   -12.019   37.390   38.496
H   -3.578   42.982   28.163
H   -3.423   43.404   21.903
H   2.885   44.240   21.902
H   0.904   42.103   29.743
H   -1.640   43.727   30.011
H   -2.226   41.852   30.973
H   -1.164   40.827   29.900
H   -2.721   41.236   29.306
H   0.098   44.949   30.452
H   1.722   44.591   30.121
H   0.292   44.517   32.528
H   1.878   43.750   32.206
H   -6.319   42.391   26.420
H   -6.679   44.117   26.545
H   -5.584   43.371   27.656
H   -7.162   42.713   24.325
H   -7.749   43.319   22.952
H   -7.692   44.456   24.215
H   -1.462   44.362   20.137
H   1.258   43.403   20.274
H   -0.614   42.244   19.219
H   -2.061   41.782   20.074
H   -0.416   41.560   20.879
H   0.569   45.430   18.927
H   2.107   45.418   19.870
H   0.423   47.556   19.893
H   1.413   47.235   21.252
H   -0.355   46.704   21.155
H   4.935   43.967   22.628
H   5.616   42.853   23.774
H   5.935   44.573   23.990
H   3.661   43.992   28.923
H   5.029   40.142   30.410
H   3.246   40.013   30.714
H   4.253   40.853   31.931
H   -0.395   40.989   34.491
H   -1.335   40.478   33.163
H   -3.422   41.149   33.772
H   -1.212   41.919   36.718
H   -1.914   40.415   37.363
H   -2.776   41.901   37.706
H   -4.665   40.900   37.091
H   -4.913   40.423   35.453
H   -5.053   43.159   35.110
H   -5.560   42.930   36.816
H   -7.098   41.042   35.084
H   -7.135   42.637   34.427
H   -7.862   42.409   37.298
H   -7.702   44.793   36.495
H   -9.408   44.352   36.383
H   -8.439   44.349   34.934
H   -9.299   40.944   35.252
H   -10.164   42.438   35.609
H   -9.059   41.025   38.048
H   -10.235   40.108   37.091
H   -10.756   43.039   37.864
H   -10.911   41.776   39.076
H   -12.339   40.890   36.589
H   -12.563   43.798   37.049
H   -14.110   42.825   36.861
H   -12.944   42.748   35.608
H   -13.825   41.861   38.981
H   -12.778   40.540   39.255
H   -14.282   39.951   36.832
H   -15.407   40.506   37.987
H   -15.202   38.305   38.661
H   -13.710   38.811   39.536
H   -12.901   38.313   36.820
H   -14.737   36.188   37.769
H   -13.566   36.093   36.465
H   -14.880   37.289   36.397
H   -11.718   36.415   38.114
H   -12.220   37.266   39.560
H   -11.212   38.115   38.392

