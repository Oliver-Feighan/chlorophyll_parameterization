%nproc=24
%mem=75GB
#p PBE1PBE/Def2SVP td=(nstates=5)

step_1851_chromophore_6 TDDFT with PBE1PBE functional

0 1
Mg   17.621   -1.816   27.677
C   16.915   0.374   30.292
C   19.593   -3.663   29.977
C   19.278   -3.498   25.161
C   16.542   0.668   25.365
N   18.195   -1.638   29.814
C   17.718   -0.709   30.676
C   18.137   -0.942   32.086
C   19.214   -2.109   31.972
C   18.995   -2.542   30.502
C   20.709   -1.661   32.310
C   16.847   -1.265   32.966
C   16.768   -0.611   34.379
C   18.033   -0.100   35.075
O   18.302   1.072   35.303
O   18.883   -1.117   35.369
N   18.930   -3.643   27.588
C   19.545   -4.213   28.667
C   20.231   -5.431   28.147
C   20.046   -5.447   26.672
C   19.342   -4.156   26.409
C   20.915   -6.436   29.062
C   20.422   -6.435   25.569
O   20.065   -6.351   24.377
C   21.258   -7.622   25.937
N   17.949   -1.398   25.594
C   18.638   -2.309   24.770
C   18.526   -1.899   23.374
C   17.402   -0.789   23.375
C   17.252   -0.437   24.873
C   19.845   -1.450   22.753
C   16.026   -1.128   22.787
C   15.416   0.092   22.078
N   16.684   0.015   27.781
C   16.259   0.857   26.735
C   15.493   1.971   27.369
C   15.810   1.812   28.754
C   16.480   0.596   28.959
C   14.619   3.119   26.808
C   15.531   2.425   30.045
O   14.928   3.480   30.300
C   16.237   1.500   31.091
C   17.183   2.375   31.966
O   18.374   2.415   31.906
O   16.405   2.946   32.970
C   17.164   3.760   33.912
C   20.243   -0.743   35.851
C   20.341   -0.362   37.272
C   21.377   -0.591   38.115
C   22.713   -1.174   37.660
C   21.317   -0.469   39.659
C   20.572   -1.587   40.414
C   21.461   -2.736   40.983
C   21.003   -4.119   40.655
C   19.524   -4.298   41.172
C   21.953   -5.158   41.239
C   23.242   -5.360   40.321
C   23.462   -6.842   39.870
C   24.898   -7.502   40.197
C   25.347   -8.342   38.998
C   24.705   -8.327   41.537
C   25.613   -7.773   42.670
C   27.042   -8.372   42.882
C   27.069   -9.468   43.980
C   27.738   -10.674   43.376
C   27.678   -9.005   45.357
H   20.318   -4.119   30.654
H   19.613   -3.964   24.233
H   16.006   1.231   24.598
H   18.576   -0.014   32.451
H   19.035   -2.977   32.607
H   20.687   -0.644   32.702
H   21.306   -1.668   31.398
H   21.165   -2.338   33.033
H   16.847   -2.355   32.980
H   15.909   -1.011   32.473
H   16.349   -1.328   35.085
H   16.073   0.224   34.293
H   21.970   -6.328   28.814
H   20.426   -7.404   28.948
H   20.797   -6.117   30.098
H   22.175   -7.193   26.339
H   21.455   -8.123   24.989
H   20.792   -8.304   26.649
H   18.167   -2.816   22.906
H   17.784   0.129   22.928
H   20.720   -1.817   23.291
H   19.881   -0.361   22.719
H   19.891   -1.977   21.800
H   15.345   -1.441   23.578
H   16.230   -1.961   22.114
H   14.609   0.457   22.714
H   15.021   -0.081   21.077
H   16.096   0.935   21.954
H   14.896   3.235   25.760
H   14.738   4.021   27.409
H   13.565   2.845   26.777
H   15.396   1.098   31.657
H   16.582   4.179   34.733
H   17.644   4.581   33.379
H   17.943   3.258   34.486
H   20.660   0.127   35.343
H   20.903   -1.595   35.684
H   19.435   -0.074   37.804
H   22.863   -2.074   38.256
H   23.454   -0.460   38.020
H   22.797   -1.216   36.574
H   20.885   0.503   39.896
H   22.277   -0.327   40.155
H   19.873   -1.915   39.645
H   20.086   -1.161   41.292
H   21.417   -2.616   42.066
H   22.503   -2.554   40.721
H   21.023   -4.077   39.566
H   18.829   -4.079   40.362
H   19.461   -3.852   42.165
H   19.488   -5.377   41.321
H   21.410   -6.058   41.528
H   22.282   -4.758   42.199
H   24.146   -5.023   40.828
H   23.229   -4.742   39.423
H   23.296   -6.908   38.795
H   22.764   -7.604   40.216
H   25.651   -6.726   40.333
H   26.155   -7.833   38.472
H   24.505   -8.579   38.348
H   25.662   -9.373   39.164
H   24.929   -9.379   41.361
H   23.704   -8.294   41.968
H   24.971   -7.778   43.551
H   25.842   -6.740   42.405
H   27.781   -7.605   43.116
H   27.318   -8.738   41.893
H   26.023   -9.769   44.041
H   28.191   -11.310   44.137
H   28.498   -10.312   42.683
H   27.089   -11.402   42.891
H   27.905   -7.941   45.424
H   28.574   -9.577   45.595
H   27.019   -9.312   46.169

