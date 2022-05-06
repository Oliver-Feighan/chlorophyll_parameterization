%nproc=24
%mem=50gb
#p ZINDO=(nstates=5)

step_151_chromophore_27 ZINDO

0 1
Mg   -5.319   24.552   27.519
C   -4.204   26.622   29.985
C   -6.146   22.175   29.960
C   -6.426   22.524   25.096
C   -4.016   26.842   25.117
N   -5.263   24.393   29.717
C   -4.968   25.511   30.480
C   -5.263   25.243   32.005
C   -5.431   23.682   31.962
C   -5.648   23.401   30.446
C   -4.336   22.830   32.578
C   -6.479   26.100   32.559
C   -6.145   26.855   33.798
C   -6.162   25.998   35.071
O   -5.266   25.534   35.702
O   -7.467   25.783   35.439
N   -6.083   22.581   27.610
C   -6.366   21.769   28.645
C   -6.845   20.500   28.175
C   -6.730   20.557   26.743
C   -6.434   21.913   26.422
C   -7.114   19.264   29.018
C   -6.851   19.439   25.718
O   -6.568   19.620   24.587
C   -7.216   18.127   26.111
N   -5.118   24.593   25.431
C   -5.753   23.644   24.654
C   -5.730   23.993   23.115
C   -5.037   25.373   23.011
C   -4.750   25.669   24.570
C   -5.109   22.827   22.246
C   -6.079   26.467   22.474
C   -6.013   26.551   20.905
N   -4.427   26.397   27.488
C   -3.884   27.175   26.498
C   -3.310   28.352   27.062
C   -3.411   28.250   28.418
C   -4.083   27.038   28.647
C   -2.710   29.423   26.213
C   -3.131   28.821   29.700
O   -2.457   29.785   30.020
C   -3.556   27.741   30.780
C   -2.356   27.475   31.636
O   -1.435   26.719   31.309
O   -2.502   28.000   32.890
C   -1.630   27.444   33.914
C   -7.697   24.771   36.561
C   -8.731   23.703   36.187
C   -9.331   22.840   37.065
C   -9.233   23.043   38.647
C   -10.224   21.667   36.721
C   -11.715   21.960   36.885
C   -12.395   21.060   37.923
C   -12.919   19.693   37.326
C   -14.357   19.695   36.927
C   -12.596   18.560   38.246
C   -11.321   17.853   37.657
C   -10.540   17.006   38.743
C   -9.061   17.286   38.690
C   -8.812   18.504   39.498
C   -8.296   16.023   39.270
C   -6.979   15.749   38.513
C   -6.166   14.625   39.193
C   -4.903   15.184   39.877
C   -3.865   15.622   38.747
C   -4.262   14.113   40.794
H   -6.387   21.440   30.730
H   -6.709   21.877   24.262
H   -3.648   27.530   24.353
H   -4.355   25.491   32.554
H   -6.325   23.385   32.511
H   -4.601   22.700   33.627
H   -3.416   23.413   32.563
H   -4.209   21.896   32.030
H   -7.465   25.637   32.582
H   -6.490   26.918   31.838
H   -6.734   27.756   33.968
H   -5.180   27.353   33.881
H   -6.314   18.531   28.919
H   -8.129   18.912   28.833
H   -7.015   19.464   30.085
H   -8.130   18.102   26.704
H   -6.433   17.694   26.733
H   -7.256   17.382   25.316
H   -6.755   23.993   22.746
H   -4.134   25.286   22.407
H   -4.970   21.877   22.764
H   -4.207   23.243   21.800
H   -5.863   22.609   21.490
H   -5.854   27.435   22.923
H   -7.079   26.144   22.762
H   -6.593   27.405   20.555
H   -6.501   25.642   20.553
H   -4.984   26.589   20.548
H   -3.564   29.605   25.561
H   -1.766   29.175   25.729
H   -2.503   30.264   26.876
H   -4.368   28.127   31.397
H   -1.826   27.778   34.933
H   -0.595   27.617   33.616
H   -1.806   26.371   33.985
H   -8.127   25.429   37.317
H   -6.812   24.345   37.034
H   -8.738   23.429   35.132
H   -8.871   22.186   39.215
H   -10.182   23.364   39.077
H   -8.502   23.801   38.928
H   -9.825   20.809   37.260
H   -10.024   21.536   35.657
H   -12.210   21.726   35.942
H   -11.950   22.999   37.115
H   -13.216   21.672   38.296
H   -11.739   20.932   38.784
H   -12.365   19.546   36.399
H   -14.371   19.547   35.847
H   -14.865   20.640   37.122
H   -14.966   18.889   37.337
H   -13.405   17.830   38.255
H   -12.459   18.896   39.273
H   -10.654   18.507   37.094
H   -11.739   17.260   36.844
H   -10.775   15.948   38.624
H   -10.912   17.093   39.764
H   -8.744   17.332   37.648
H   -9.616   18.672   40.215
H   -8.689   19.325   38.791
H   -7.907   18.519   40.105
H   -8.959   15.166   39.152
H   -8.028   16.151   40.318
H   -6.579   16.760   38.444
H   -7.183   15.372   37.510
H   -5.797   13.901   38.465
H   -6.800   14.064   39.879
H   -4.996   16.138   40.396
H   -3.603   16.679   38.716
H   -4.276   15.290   37.794
H   -2.873   15.207   38.925
H   -5.071   13.485   41.167
H   -3.840   14.542   41.703
H   -3.464   13.559   40.301

