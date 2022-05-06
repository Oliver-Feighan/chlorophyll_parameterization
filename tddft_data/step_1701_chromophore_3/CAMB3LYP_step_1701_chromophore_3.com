%nproc=24
%mem=75GB
#p cam-b3lyp/Def2SVP td=(nstates=5)

step_1701_chromophore_3 TDDFT with cam-b3lyp functional

0 1
Mg   0.995   7.755   25.901
C   1.835   9.843   28.522
C   1.524   5.018   28.121
C   1.217   5.611   23.317
C   1.166   10.463   23.770
N   1.552   7.509   28.065
C   1.850   8.547   28.976
C   1.894   7.990   30.352
C   2.124   6.446   30.119
C   1.704   6.300   28.679
C   3.600   6.044   30.462
C   0.619   8.379   31.085
C   0.878   8.813   32.566
C   2.094   8.146   33.174
O   3.219   8.488   32.969
O   1.785   7.015   33.857
N   1.114   5.684   25.755
C   1.240   4.720   26.744
C   1.269   3.315   26.186
C   1.306   3.466   24.717
C   1.212   4.936   24.534
C   1.404   2.017   26.967
C   1.496   2.416   23.678
O   1.485   2.637   22.476
C   1.462   0.998   23.998
N   1.084   8.003   23.896
C   1.155   6.973   23.039
C   1.093   7.530   21.514
C   0.802   8.993   21.768
C   0.961   9.225   23.243
C   2.466   7.306   20.847
C   -0.664   9.493   21.291
C   -0.621   10.929   20.771
N   1.232   9.799   26.085
C   1.355   10.751   25.115
C   1.533   12.056   25.788
C   1.727   11.728   27.160
C   1.617   10.356   27.248
C   1.691   13.463   25.161
C   2.062   12.303   28.443
O   2.393   13.445   28.818
C   1.924   11.141   29.412
C   3.043   11.128   30.387
O   4.166   10.659   30.227
O   2.668   11.931   31.428
C   3.727   12.004   32.492
C   2.842   6.251   34.409
C   2.199   5.106   35.179
C   2.610   4.662   36.356
C   3.914   5.063   37.054
C   1.791   3.671   36.957
C   2.185   2.209   37.327
C   1.095   1.183   36.720
C   0.325   0.429   37.847
C   -1.203   0.865   37.835
C   0.446   -1.101   37.797
C   1.068   -1.903   38.959
C   2.563   -1.842   39.048
C   3.210   -3.062   39.749
C   4.352   -2.455   40.701
C   3.701   -4.202   38.876
C   4.027   -5.570   39.599
C   2.944   -6.677   39.245
C   3.613   -7.906   38.598
C   2.492   -8.818   37.985
C   4.556   -8.688   39.555
H   1.634   4.158   28.786
H   1.268   5.032   22.392
H   1.116   11.363   23.154
H   2.829   8.396   30.738
H   1.519   5.870   30.820
H   4.174   6.890   30.839
H   4.068   5.731   29.528
H   3.525   5.163   31.099
H   -0.086   7.556   31.203
H   0.126   9.145   30.486
H   0.038   8.595   33.227
H   1.152   9.863   32.474
H   0.707   1.230   26.680
H   1.028   2.272   27.958
H   2.478   1.838   27.009
H   2.316   0.792   24.644
H   1.436   0.322   23.143
H   0.500   0.682   24.401
H   0.319   6.970   20.989
H   1.544   9.616   21.268
H   2.514   6.297   20.438
H   3.389   7.473   21.402
H   2.508   8.005   20.011
H   -1.394   9.449   22.099
H   -0.943   8.730   20.566
H   0.386   11.334   20.875
H   -1.280   11.613   21.305
H   -0.896   10.926   19.716
H   1.164   13.451   24.207
H   2.718   13.828   25.186
H   1.128   14.213   25.717
H   1.019   11.331   29.990
H   3.159   12.388   33.339
H   4.578   12.669   32.348
H   4.112   11.031   32.798
H   3.538   6.890   34.952
H   3.504   5.939   33.601
H   1.243   4.801   34.752
H   3.721   5.478   38.043
H   4.619   5.774   36.623
H   4.521   4.160   37.113
H   0.752   3.701   36.627
H   1.733   4.074   37.968
H   2.361   2.066   38.394
H   3.046   1.948   36.712
H   1.615   0.432   36.126
H   0.474   1.668   35.967
H   0.720   0.764   38.806
H   -1.253   1.482   38.732
H   -1.793   0.008   38.160
H   -1.649   1.529   37.095
H   0.972   -1.541   36.950
H   -0.556   -1.504   37.651
H   0.709   -2.932   38.958
H   0.624   -1.479   39.859
H   2.803   -0.953   39.631
H   2.982   -1.695   38.053
H   2.436   -3.438   40.418
H   4.055   -2.714   41.717
H   4.635   -1.405   40.629
H   5.262   -2.951   40.364
H   4.594   -3.840   38.367
H   2.925   -4.483   38.163
H   4.065   -5.367   40.669
H   4.966   -6.010   39.263
H   2.166   -6.349   38.555
H   2.506   -7.130   40.134
H   4.283   -7.512   37.834
H   2.402   -8.521   36.940
H   1.564   -8.583   38.507
H   2.742   -9.877   38.040
H   5.583   -8.852   39.229
H   4.147   -9.682   39.731
H   4.738   -8.123   40.470

