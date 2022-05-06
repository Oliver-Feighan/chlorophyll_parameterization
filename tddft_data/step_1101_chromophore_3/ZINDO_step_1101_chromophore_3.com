%nproc=24
%mem=50gb
#p ZINDO=(nstates=5)

step_1101_chromophore_3 ZINDO

0 1
Mg   1.695   8.620   26.341
C   1.903   10.775   29.071
C   2.156   6.023   28.234
C   2.336   6.799   23.448
C   1.787   11.539   24.280
N   2.113   8.473   28.476
C   1.899   9.412   29.433
C   1.914   8.737   30.869
C   2.168   7.323   30.565
C   2.221   7.227   28.981
C   3.415   6.633   31.292
C   0.600   9.115   31.684
C   0.478   9.215   33.177
C   1.611   8.707   34.088
O   2.582   9.333   34.461
O   1.369   7.400   34.392
N   2.092   6.658   25.927
C   2.133   5.720   26.884
C   2.222   4.414   26.253
C   2.413   4.651   24.837
C   2.278   6.111   24.681
C   2.269   3.087   27.055
C   2.579   3.610   23.744
O   2.497   3.870   22.541
C   2.763   2.256   24.087
N   1.982   9.098   24.223
C   2.056   8.134   23.237
C   2.000   8.678   21.781
C   1.682   10.194   22.106
C   1.809   10.297   23.617
C   3.256   8.237   21.013
C   0.307   10.747   21.619
C   0.215   11.947   20.642
N   1.823   10.738   26.570
C   1.853   11.748   25.662
C   1.838   13.025   26.355
C   1.952   12.674   27.746
C   1.940   11.292   27.790
C   1.982   14.419   25.745
C   2.126   13.245   29.105
O   2.355   14.337   29.579
C   2.238   11.947   30.000
C   3.529   11.922   30.692
O   4.625   11.839   30.128
O   3.462   12.061   32.082
C   4.587   11.704   32.828
C   2.130   6.794   35.474
C   1.304   5.932   36.321
C   1.565   5.283   37.471
C   2.933   5.483   38.159
C   0.582   4.396   38.201
C   0.770   2.882   37.858
C   0.783   2.002   39.107
C   2.022   1.092   39.029
C   3.348   1.737   39.377
C   1.836   -0.098   40.065
C   2.290   -1.457   39.352
C   3.079   -2.398   40.370
C   4.663   -2.435   40.082
C   5.169   -3.867   39.678
C   5.502   -1.680   41.112
C   6.352   -0.568   40.505
C   7.854   -0.944   40.084
C   8.820   0.244   40.506
C   9.885   0.429   39.407
C   9.584   -0.080   41.861
H   2.158   5.143   28.881
H   2.328   6.182   22.547
H   1.632   12.430   23.667
H   2.831   9.158   31.280
H   1.256   6.802   30.857
H   2.959   5.954   32.012
H   4.116   7.284   31.814
H   4.070   6.036   30.658
H   -0.240   8.538   31.299
H   0.341   10.112   31.328
H   -0.492   8.808   33.462
H   0.573   10.270   33.433
H   1.850   3.355   28.024
H   3.249   2.610   27.030
H   1.621   2.368   26.553
H   2.996   1.566   23.276
H   1.877   1.910   24.620
H   3.609   2.220   24.774
H   1.135   8.104   21.449
H   2.549   10.607   21.589
H   3.980   7.626   21.551
H   3.800   9.051   20.534
H   2.832   7.592   20.244
H   -0.122   11.238   22.492
H   -0.409   9.976   21.335
H   1.184   12.021   20.149
H   0.199   12.938   21.095
H   -0.516   11.934   19.834
H   1.667   14.528   24.707
H   3.053   14.620   25.786
H   1.454   15.070   26.440
H   1.324   12.121   30.569
H   5.500   12.297   32.776
H   4.920   10.697   32.577
H   4.299   11.747   33.878
H   2.708   7.524   36.040
H   2.784   6.012   35.087
H   0.280   5.771   35.981
H   3.561   6.217   37.654
H   3.398   4.498   38.202
H   2.667   5.737   39.185
H   -0.434   4.691   37.936
H   0.828   4.621   39.238
H   1.709   2.751   37.321
H   -0.053   2.604   37.200
H   -0.095   1.364   39.210
H   0.805   2.683   39.958
H   2.023   0.704   38.010
H   4.048   1.532   38.567
H   3.824   1.357   40.281
H   3.123   2.792   39.533
H   0.818   -0.227   40.434
H   2.379   0.163   40.973
H   2.732   -1.352   38.361
H   1.380   -1.985   39.067
H   2.594   -3.372   40.309
H   3.051   -1.989   41.381
H   4.776   -1.902   39.138
H   6.114   -3.979   40.210
H   5.250   -4.032   38.604
H   4.594   -4.742   39.981
H   6.163   -2.382   41.620
H   4.885   -1.210   41.877
H   6.373   0.232   41.245
H   5.857   -0.044   39.688
H   7.888   -1.079   39.002
H   8.182   -1.929   40.416
H   8.294   1.188   40.652
H   9.480   0.101   38.450
H   10.785   -0.136   39.651
H   10.194   1.469   39.299
H   9.587   0.858   42.416
H   10.629   -0.373   41.758
H   9.063   -0.849   42.431

