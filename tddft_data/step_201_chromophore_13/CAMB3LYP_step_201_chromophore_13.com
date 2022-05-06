%nproc=24
%mem=75GB
#p cam-b3lyp/Def2SVP td=(nstates=5)

step_201_chromophore_13 TDDFT with cam-b3lyp functional

0 1
Mg   46.460   24.208   28.504
C   46.899   26.475   31.036
C   45.164   21.948   30.794
C   46.619   21.697   26.116
C   48.024   26.375   26.289
N   46.066   24.268   30.642
C   46.273   25.333   31.432
C   45.719   25.086   32.809
C   45.404   23.531   32.802
C   45.547   23.248   31.319
C   46.307   22.638   33.732
C   44.493   25.990   33.291
C   44.715   26.761   34.666
C   43.886   26.396   35.916
O   42.688   26.179   35.809
O   44.605   26.675   36.984
N   45.844   22.201   28.416
C   45.361   21.461   29.506
C   44.968   20.106   29.061
C   45.450   20.048   27.733
C   46.021   21.396   27.326
C   44.214   19.141   29.886
C   45.274   18.838   26.737
O   45.551   18.967   25.535
C   44.715   17.473   27.208
N   47.242   23.972   26.507
C   47.150   22.876   25.703
C   47.901   23.040   24.357
C   48.134   24.632   24.370
C   47.695   25.076   25.770
C   49.267   22.119   24.307
C   47.481   25.395   23.285
C   45.941   25.639   23.334
N   47.300   26.099   28.562
C   47.778   26.919   27.565
C   48.154   28.253   28.129
C   47.848   28.070   29.511
C   47.338   26.772   29.707
C   48.670   29.449   27.430
C   47.843   28.676   30.754
O   48.270   29.799   31.053
C   47.203   27.672   31.798
C   48.376   27.454   32.767
O   49.540   27.193   32.406
O   47.956   27.383   34.017
C   48.990   27.129   35.056
C   43.769   26.711   38.192
C   43.956   25.436   38.904
C   43.472   25.065   40.092
C   42.853   26.023   41.029
C   43.780   23.639   40.614
C   42.608   22.721   40.495
C   42.030   22.421   41.883
C   40.669   21.692   41.835
C   40.549   20.710   40.635
C   39.507   22.707   41.869
C   38.809   22.464   43.303
C   38.250   23.783   43.815
C   38.343   23.955   45.356
C   37.112   24.496   46.062
C   39.521   24.883   45.788
C   40.887   24.296   45.458
C   41.747   24.351   46.730
C   43.237   24.354   46.328
C   43.776   23.087   45.559
C   44.106   24.538   47.616
H   44.755   21.236   31.513
H   46.895   20.908   25.413
H   48.573   27.049   25.628
H   46.483   25.313   33.554
H   44.400   23.282   33.146
H   46.887   23.302   34.372
H   46.886   21.928   33.141
H   45.687   21.989   34.350
H   43.639   25.334   33.456
H   44.303   26.776   32.560
H   44.625   27.843   34.576
H   45.768   26.664   34.930
H   44.842   18.254   29.974
H   43.289   18.889   29.367
H   43.856   19.520   30.843
H   44.953   16.778   26.402
H   43.634   17.572   27.107
H   44.875   17.122   28.227
H   47.286   22.681   23.532
H   49.205   24.760   24.209
H   49.216   21.361   25.088
H   50.142   22.751   24.463
H   49.214   21.610   23.345
H   47.685   24.819   22.382
H   48.034   26.332   23.220
H   45.823   26.699   23.561
H   45.395   25.188   24.163
H   45.386   25.377   22.434
H   47.909   30.220   27.310
H   49.204   29.207   26.511
H   49.295   29.948   28.171
H   46.344   28.148   32.271
H   49.476   26.177   34.840
H   48.363   27.121   35.947
H   49.739   27.918   35.130
H   42.710   26.908   38.028
H   44.185   27.496   38.824
H   44.542   24.686   38.373
H   43.526   26.043   41.887
H   41.830   25.693   41.209
H   42.823   27.079   40.762
H   44.170   23.699   41.630
H   44.593   23.246   40.005
H   42.951   21.847   39.941
H   41.890   23.342   39.959
H   42.063   23.257   42.582
H   42.709   21.714   42.361
H   40.570   21.065   42.721
H   39.656   20.085   40.617
H   41.448   20.093   40.665
H   40.597   21.191   39.658
H   38.801   22.373   41.109
H   39.858   23.729   41.725
H   39.390   21.990   44.094
H   37.986   21.791   43.063
H   37.174   23.731   43.651
H   38.619   24.616   43.218
H   38.564   23.034   45.894
H   37.133   24.083   47.070
H   36.177   24.294   45.539
H   37.173   25.583   46.014
H   39.366   25.296   46.784
H   39.443   25.637   45.004
H   41.274   24.875   44.620
H   40.774   23.255   45.155
H   41.506   23.359   47.112
H   41.578   25.080   47.523
H   43.500   25.193   45.684
H   43.697   22.230   46.229
H   44.840   23.292   45.449
H   43.209   22.984   44.634
H   43.695   24.004   48.473
H   43.863   25.578   47.835
H   45.168   24.414   47.405
