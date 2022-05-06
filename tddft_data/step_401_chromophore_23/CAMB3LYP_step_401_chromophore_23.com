%nproc=24
%mem=75GB
#p cam-b3lyp/Def2SVP td=(nstates=5)

step_401_chromophore_23 TDDFT with cam-b3lyp functional

0 1
Mg   -10.005   40.063   43.152
C   -8.693   37.115   42.125
C   -7.878   41.741   41.022
C   -11.915   42.954   43.400
C   -12.674   38.335   44.618
N   -8.605   39.454   41.566
C   -8.096   38.189   41.415
C   -6.984   38.088   40.422
C   -6.666   39.625   40.150
C   -7.729   40.357   40.977
C   -5.184   40.097   40.561
C   -7.495   37.338   39.162
C   -6.656   36.186   38.580
C   -5.564   36.620   37.585
O   -4.603   37.354   37.799
O   -6.009   36.232   36.298
N   -9.801   42.170   42.482
C   -8.781   42.629   41.632
C   -9.010   44.029   41.402
C   -10.203   44.434   42.079
C   -10.721   43.144   42.747
C   -8.212   44.902   40.419
C   -10.837   45.837   41.937
O   -10.253   46.667   41.228
C   -12.066   46.330   42.664
N   -12.028   40.656   43.850
C   -12.534   41.831   43.921
C   -14.020   41.717   44.467
C   -14.201   40.271   44.811
C   -12.876   39.680   44.412
C   -14.303   42.659   45.701
C   -15.391   39.566   44.086
C   -16.238   38.560   44.918
N   -10.491   38.056   43.544
C   -11.580   37.507   44.135
C   -11.423   36.052   44.203
C   -10.162   35.830   43.405
C   -9.750   37.117   43.007
C   -12.395   35.043   44.761
C   -9.248   34.822   42.825
O   -9.251   33.627   42.857
C   -8.111   35.639   42.161
C   -6.892   35.569   43.040
O   -6.526   36.231   43.996
O   -6.286   34.363   42.785
C   -4.981   34.184   43.406
C   -5.482   36.890   35.092
C   -6.569   36.825   33.985
C   -7.687   37.610   33.958
C   -8.026   38.619   35.064
C   -8.723   37.553   32.812
C   -8.287   38.262   31.451
C   -8.068   37.224   30.316
C   -6.798   37.589   29.430
C   -6.511   36.411   28.417
C   -6.981   38.844   28.643
C   -6.329   40.117   29.210
C   -7.375   41.288   29.289
C   -6.817   42.432   30.157
C   -7.243   42.368   31.654
C   -6.888   43.870   29.491
C   -5.570   44.639   29.756
C   -4.563   44.807   28.610
C   -3.914   46.209   28.325
C   -3.694   46.468   26.779
C   -2.716   46.462   29.206
H   -7.072   42.372   40.642
H   -12.593   43.808   43.340
H   -13.458   37.813   45.171
H   -6.144   37.522   40.827
H   -6.810   39.865   39.096
H   -5.321   40.349   41.612
H   -4.762   40.983   40.088
H   -4.506   39.252   40.441
H   -7.611   38.015   38.316
H   -8.477   36.907   39.359
H   -7.149   35.271   38.252
H   -5.980   35.881   39.379
H   -8.724   45.148   39.488
H   -7.299   44.411   40.081
H   -7.798   45.781   40.913
H   -12.289   47.384   42.494
H   -11.897   46.204   43.733
H   -12.927   45.807   42.248
H   -14.596   42.051   43.604
H   -14.254   40.041   45.875
H   -14.851   42.095   46.456
H   -14.973   43.477   45.438
H   -13.402   43.153   46.063
H   -14.987   39.086   43.194
H   -16.118   40.363   43.932
H   -15.596   38.029   45.621
H   -16.752   37.817   44.307
H   -17.064   39.056   45.427
H   -12.300   34.943   45.842
H   -12.402   34.119   44.182
H   -13.427   35.369   44.631
H   -7.964   35.387   41.111
H   -5.034   33.443   44.204
H   -4.613   35.136   43.788
H   -4.390   33.825   42.563
H   -4.635   36.266   34.807
H   -5.219   37.943   35.195
H   -6.375   36.213   33.104
H   -7.293   38.541   35.867
H   -9.023   38.353   35.415
H   -8.085   39.605   34.604
H   -9.595   38.032   33.257
H   -8.979   36.498   32.710
H   -7.359   38.746   31.757
H   -9.075   38.938   31.120
H   -8.936   37.338   29.667
H   -8.056   36.220   30.740
H   -5.952   37.674   30.112
H   -5.518   35.998   28.597
H   -6.854   36.674   27.416
H   -7.160   35.578   28.685
H   -8.038   39.021   28.440
H   -6.567   38.749   27.639
H   -5.597   40.323   28.429
H   -5.763   40.039   30.138
H   -8.286   40.917   29.759
H   -7.641   41.670   28.304
H   -5.774   42.141   30.279
H   -7.991   41.588   31.800
H   -7.582   43.360   31.952
H   -6.372   42.078   32.242
H   -7.755   44.400   29.885
H   -7.162   43.850   28.436
H   -5.077   44.303   30.667
H   -5.878   45.613   30.135
H   -4.959   44.246   27.764
H   -3.753   44.137   28.900
H   -4.678   46.915   28.649
H   -4.548   46.128   26.193
H   -2.810   45.862   26.575
H   -3.381   47.511   26.728
H   -2.001   46.983   28.569
H   -2.249   45.507   29.448
H   -2.893   47.210   29.978

