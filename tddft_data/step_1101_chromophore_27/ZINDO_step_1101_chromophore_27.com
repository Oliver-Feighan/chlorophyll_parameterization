%nproc=24
%mem=50gb
#p ZINDO=(nstates=5)

step_1101_chromophore_27 ZINDO

0 1
Mg   -5.663   24.994   27.721
C   -3.781   26.862   30.112
C   -6.648   22.843   30.133
C   -7.163   23.011   25.263
C   -4.199   26.833   25.232
N   -5.180   24.879   29.773
C   -4.535   25.789   30.643
C   -4.838   25.543   32.129
C   -5.671   24.154   32.071
C   -5.799   23.855   30.567
C   -5.032   23.031   32.851
C   -5.660   26.698   32.741
C   -5.341   27.007   34.171
C   -6.399   26.716   35.211
O   -7.349   27.404   35.501
O   -6.050   25.682   35.983
N   -6.857   23.177   27.755
C   -7.123   22.461   28.828
C   -7.771   21.243   28.412
C   -7.910   21.192   27.013
C   -7.241   22.457   26.619
C   -8.271   20.262   29.456
C   -8.577   20.129   26.121
O   -8.584   20.174   24.870
C   -9.351   19.025   26.800
N   -5.684   24.890   25.581
C   -6.412   24.046   24.788
C   -6.308   24.331   23.275
C   -5.313   25.613   23.300
C   -5.085   25.853   24.801
C   -5.739   23.147   22.488
C   -5.933   26.929   22.632
C   -5.024   27.737   21.787
N   -4.284   26.551   27.659
C   -3.908   27.221   26.574
C   -2.898   28.210   27.007
C   -2.908   28.200   28.372
C   -3.651   27.092   28.735
C   -2.300   29.219   26.140
C   -2.369   28.744   29.602
O   -1.591   29.685   29.766
C   -3.005   27.962   30.788
C   -1.831   27.506   31.619
O   -0.795   26.975   31.262
O   -2.118   27.902   32.885
C   -0.974   27.720   33.798
C   -6.868   25.421   37.225
C   -7.775   24.223   36.959
C   -8.361   23.256   37.686
C   -7.982   23.042   39.121
C   -9.317   22.177   37.154
C   -10.716   22.078   37.819
C   -10.968   20.767   38.660
C   -12.053   19.830   38.112
C   -13.422   20.496   38.493
C   -11.849   18.387   38.836
C   -11.174   17.446   37.872
C   -10.316   16.436   38.720
C   -8.806   16.683   38.699
C   -8.463   17.877   39.627
C   -8.075   15.364   39.262
C   -6.784   15.097   38.372
C   -5.495   15.611   39.125
C   -4.491   14.402   39.349
C   -4.972   13.419   40.520
C   -3.153   15.132   39.797
H   -6.921   22.128   30.912
H   -7.593   22.323   24.533
H   -3.868   27.556   24.484
H   -3.879   25.367   32.617
H   -6.673   24.213   32.497
H   -5.784   22.374   33.288
H   -4.363   23.333   33.656
H   -4.461   22.415   32.157
H   -6.675   26.308   32.665
H   -5.785   27.600   32.143
H   -5.102   28.068   34.248
H   -4.450   26.444   34.451
H   -9.345   20.090   29.385
H   -7.998   20.540   30.474
H   -7.640   19.386   29.308
H   -9.959   18.511   26.055
H   -10.099   19.384   27.508
H   -8.539   18.526   27.329
H   -7.306   24.513   22.876
H   -4.321   25.337   22.942
H   -4.946   23.462   21.809
H   -6.530   22.633   21.941
H   -5.348   22.395   23.172
H   -6.271   27.583   23.436
H   -6.837   26.677   22.079
H   -5.491   27.889   20.814
H   -4.033   27.287   21.724
H   -4.814   28.724   22.198
H   -1.758   28.621   25.408
H   -1.436   29.673   26.626
H   -2.985   29.961   25.728
H   -3.704   28.595   31.334
H   -0.078   28.135   33.336
H   -0.689   26.672   33.891
H   -1.168   28.138   34.786
H   -7.354   26.328   37.584
H   -6.235   25.166   38.075
H   -8.044   24.178   35.904
H   -7.118   23.671   39.334
H   -7.518   22.060   39.218
H   -8.797   23.188   39.830
H   -8.733   21.273   37.327
H   -9.485   22.191   36.078
H   -11.376   22.207   36.961
H   -10.842   23.007   38.374
H   -11.163   21.182   39.649
H   -10.075   20.154   38.779
H   -11.913   19.802   37.031
H   -13.865   20.798   37.544
H   -13.367   21.424   39.061
H   -14.044   19.705   38.911
H   -12.773   17.998   39.264
H   -11.224   18.555   39.712
H   -10.535   17.905   37.118
H   -11.889   16.873   37.283
H   -10.632   15.470   38.326
H   -10.594   16.424   39.774
H   -8.453   16.952   37.704
H   -9.279   17.999   40.338
H   -8.528   18.766   39.000
H   -7.457   17.757   40.029
H   -8.746   14.508   39.191
H   -7.846   15.478   40.322
H   -6.605   15.669   37.461
H   -6.752   14.051   38.067
H   -5.725   15.950   40.135
H   -5.046   16.468   38.624
H   -4.434   13.895   38.385
H   -5.971   13.023   40.341
H   -4.875   13.950   41.467
H   -4.314   12.550   40.493
H   -2.371   15.069   39.040
H   -2.716   14.695   40.694
H   -3.288   16.194   40.004

