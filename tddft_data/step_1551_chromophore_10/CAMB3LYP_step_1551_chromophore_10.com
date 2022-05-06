%nproc=24
%mem=75GB
#p cam-b3lyp/Def2SVP td=(nstates=5)

step_1551_chromophore_10 TDDFT with cam-b3lyp functional

0 1
Mg   41.221   7.390   29.128
C   43.129   8.937   31.702
C   39.237   5.976   31.550
C   39.920   5.617   26.804
C   43.716   8.511   26.899
N   41.088   7.664   31.380
C   42.048   8.226   32.194
C   41.509   8.152   33.631
C   40.294   7.184   33.583
C   40.243   6.860   32.044
C   40.359   5.969   34.492
C   41.135   9.547   34.121
C   40.131   9.649   35.277
C   40.430   10.635   36.408
O   39.635   11.496   36.803
O   41.730   10.436   36.899
N   39.737   6.071   29.178
C   39.014   5.566   30.217
C   37.893   4.743   29.703
C   38.031   4.753   28.305
C   39.294   5.389   28.023
C   36.829   4.052   30.619
C   37.109   4.079   27.200
O   37.392   4.024   25.990
C   35.704   3.690   27.694
N   41.958   6.822   27.221
C   41.067   6.265   26.408
C   41.484   6.345   24.933
C   42.530   7.461   24.937
C   42.779   7.645   26.444
C   42.015   5.035   24.409
C   42.116   8.772   24.287
C   40.769   9.382   24.675
N   43.034   8.569   29.200
C   43.981   8.800   28.216
C   45.150   9.519   28.884
C   44.760   9.662   30.197
C   43.540   8.989   30.348
C   46.418   9.920   28.282
C   45.326   10.138   31.423
O   46.409   10.674   31.714
C   44.273   9.630   32.502
C   44.887   8.800   33.617
O   45.338   7.658   33.575
O   45.009   9.605   34.678
C   45.684   8.996   35.849
C   42.054   11.115   38.136
C   41.182   10.615   39.314
C   41.284   10.942   40.650
C   42.249   11.970   41.280
C   40.351   10.274   41.616
C   39.131   11.215   41.886
C   37.883   10.446   42.307
C   37.710   10.553   43.869
C   37.821   9.176   44.632
C   36.469   11.399   44.123
C   36.473   12.135   45.504
C   35.159   12.038   46.310
C   34.789   13.400   47.092
C   35.243   13.245   48.534
C   33.284   13.830   46.943
C   33.020   14.514   45.609
C   31.730   15.345   45.769
C   31.824   16.596   44.747
C   32.014   17.912   45.591
C   30.629   16.757   43.741
H   38.597   5.603   32.352
H   39.269   5.267   25.999
H   44.429   8.871   26.153
H   42.233   7.742   34.335
H   39.360   7.739   33.672
H   40.683   5.112   33.901
H   39.398   5.801   34.979
H   41.141   6.138   35.232
H   40.692   10.056   33.265
H   42.062   10.041   34.415
H   40.119   8.678   35.773
H   39.129   9.774   34.866
H   36.826   2.997   30.342
H   35.815   4.390   30.409
H   37.044   4.130   31.685
H   35.137   3.552   26.773
H   35.339   4.540   28.271
H   35.729   2.776   28.286
H   40.640   6.752   24.375
H   43.446   7.196   24.408
H   41.486   4.185   24.841
H   43.098   4.942   24.488
H   41.814   5.056   23.338
H   42.017   8.516   23.232
H   42.853   9.531   24.545
H   40.162   8.685   25.253
H   40.212   9.656   23.779
H   40.852   10.169   25.425
H   46.642   10.983   28.365
H   46.395   9.555   27.255
H   47.117   9.274   28.812
H   43.925   10.588   32.888
H   45.593   7.910   35.854
H   45.286   9.382   36.788
H   46.751   9.222   35.850
H   42.040   12.205   38.124
H   43.037   10.782   38.469
H   40.208   10.150   39.163
H   42.912   11.569   42.046
H   41.639   12.819   41.589
H   42.773   12.424   40.439
H   40.757   10.117   42.615
H   40.007   9.252   41.457
H   38.840   11.737   40.974
H   39.393   11.949   42.648
H   38.055   9.413   42.006
H   37.051   10.797   41.697
H   38.509   11.134   44.329
H   38.703   9.032   45.256
H   37.916   8.365   43.910
H   36.953   8.928   45.244
H   35.526   10.854   44.122
H   36.420   12.218   43.405
H   36.706   13.184   45.324
H   37.262   11.702   46.119
H   35.331   11.232   47.024
H   34.246   11.774   45.777
H   35.397   14.196   46.660
H   35.302   12.226   48.917
H   34.602   13.823   49.200
H   36.275   13.567   48.676
H   32.849   14.172   47.882
H   32.774   12.892   46.728
H   32.829   13.647   44.977
H   33.945   14.965   45.251
H   31.590   15.694   46.792
H   30.821   14.822   45.471
H   32.721   16.503   44.135
H   32.457   18.732   45.026
H   32.557   17.684   46.508
H   31.039   18.306   45.878
H   29.673   16.733   44.263
H   30.549   15.816   43.197
H   30.781   17.679   43.179

