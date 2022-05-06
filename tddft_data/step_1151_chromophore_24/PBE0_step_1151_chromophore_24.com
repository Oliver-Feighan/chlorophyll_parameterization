%nproc=24
%mem=75GB
#p PBE1PBE/Def2SVP td=(nstates=5)

step_1151_chromophore_24 TDDFT with PBE1PBE functional

0 1
Mg   0.095   43.857   24.697
C   2.098   43.240   27.454
C   -2.498   42.567   26.527
C   -1.679   43.986   21.783
C   3.006   44.414   22.893
N   -0.227   43.196   26.876
C   0.707   43.088   27.843
C   0.101   43.020   29.260
C   -1.360   42.652   28.911
C   -1.425   42.806   27.353
C   -1.894   41.279   29.485
C   0.262   44.272   30.167
C   0.566   43.983   31.628
C   0.510   42.471   32.130
O   1.311   41.583   31.994
O   -0.743   42.259   32.688
N   -1.774   43.350   24.201
C   -2.697   42.884   25.114
C   -4.015   42.697   24.385
C   -3.830   43.217   23.100
C   -2.376   43.607   22.951
C   -5.198   42.193   25.136
C   -4.825   43.412   21.929
O   -4.506   43.859   20.783
C   -6.311   43.033   22.189
N   0.579   44.172   22.714
C   -0.276   44.362   21.647
C   0.389   44.561   20.270
C   1.893   44.944   20.721
C   1.844   44.454   22.216
C   0.258   43.380   19.265
C   2.174   46.438   20.660
C   1.928   47.256   19.355
N   2.131   43.975   25.066
C   3.221   44.105   24.253
C   4.443   44.019   25.001
C   4.030   43.562   26.235
C   2.635   43.595   26.228
C   5.816   44.225   24.378
C   4.498   43.138   27.497
O   5.670   42.955   27.929
C   3.231   43.121   28.426
C   3.388   41.903   29.357
O   3.108   40.720   29.054
O   3.962   42.205   30.533
C   4.703   41.072   31.223
C   -1.067   41.043   33.407
C   -2.483   41.230   33.983
C   -2.939   41.153   35.230
C   -2.163   40.693   36.385
C   -4.483   41.248   35.582
C   -4.810   42.403   36.550
C   -5.883   43.321   35.973
C   -6.644   44.107   37.122
C   -5.874   45.478   37.546
C   -8.134   44.380   36.741
C   -9.099   43.182   37.161
C   -10.087   42.886   36.031
C   -10.383   41.394   35.967
C   -9.374   40.684   35.071
C   -11.822   41.199   35.520
C   -12.910   40.918   36.690
C   -12.985   39.393   37.030
C   -13.187   39.128   38.581
C   -12.866   37.660   38.962
C   -14.685   39.555   39.020
H   -3.385   42.445   27.153
H   -2.196   44.066   20.824
H   3.850   44.670   22.249
H   0.698   42.200   29.661
H   -2.035   43.403   29.322
H   -2.503   40.762   28.743
H   -2.432   41.448   30.418
H   -1.061   40.593   29.639
H   -0.734   44.712   30.208
H   0.978   45.017   29.820
H   -0.116   44.622   32.189
H   1.612   44.257   31.768
H   -5.762   41.563   24.448
H   -5.790   43.078   25.367
H   -5.109   41.596   26.043
H   -6.457   41.972   22.393
H   -6.904   43.232   21.296
H   -6.701   43.699   22.958
H   -0.049   45.468   19.853
H   2.649   44.404   20.150
H   -0.685   43.544   18.744
H   0.142   42.501   19.899
H   1.154   43.284   18.652
H   3.191   46.578   21.026
H   1.544   46.918   21.409
H   1.114   47.973   19.469
H   1.637   46.628   18.513
H   2.831   47.798   19.073
H   5.799   44.077   23.298
H   6.463   43.486   24.852
H   6.188   45.238   24.530
H   3.122   43.917   29.163
H   5.387   40.662   30.480
H   4.126   40.326   31.768
H   5.390   41.469   31.970
H   -0.215   40.863   34.062
H   -1.074   40.215   32.698
H   -3.289   41.406   33.270
H   -2.595   39.838   36.903
H   -2.027   41.403   37.201
H   -1.162   40.417   36.051
H   -4.761   40.282   36.005
H   -5.000   41.371   34.630
H   -3.829   42.854   36.702
H   -5.073   41.756   37.386
H   -6.593   42.829   35.309
H   -5.313   44.011   35.351
H   -6.623   43.553   38.061
H   -6.541   46.340   37.541
H   -5.154   45.689   36.755
H   -5.415   45.404   38.532
H   -8.095   44.597   35.673
H   -8.504   45.333   37.119
H   -9.650   43.379   38.081
H   -8.431   42.344   37.359
H   -9.781   43.303   35.072
H   -11.013   43.428   36.220
H   -10.339   40.896   36.935
H   -9.806   40.564   34.078
H   -9.025   39.717   35.435
H   -8.470   41.256   34.863
H   -11.739   40.394   34.789
H   -12.176   42.091   35.003
H   -13.847   41.130   36.175
H   -12.682   41.435   37.622
H   -12.140   38.849   36.608
H   -13.861   38.948   36.558
H   -12.528   39.770   39.166
H   -13.641   36.992   38.587
H   -12.846   37.761   40.047
H   -11.868   37.444   38.580
H   -14.587   40.202   39.891
H   -15.245   38.657   39.282
H   -15.222   40.179   38.305

