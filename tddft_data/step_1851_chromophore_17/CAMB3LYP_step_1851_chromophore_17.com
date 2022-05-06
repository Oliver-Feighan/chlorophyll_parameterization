%nproc=24
%mem=75GB
#p cam-b3lyp/Def2SVP td=(nstates=5)

step_1851_chromophore_17 TDDFT with cam-b3lyp functional

0 1
Mg   29.670   59.687   40.501
C   26.470   58.597   39.941
C   30.774   56.738   38.964
C   32.932   60.874   40.604
C   28.435   62.759   41.472
N   28.700   57.908   39.515
C   27.365   57.650   39.505
C   27.030   56.215   38.895
C   28.519   55.573   38.869
C   29.409   56.853   39.021
C   28.712   54.368   39.856
C   26.297   56.134   37.513
C   25.458   54.866   37.138
C   25.962   54.283   35.863
O   26.501   53.249   35.703
O   25.624   55.210   34.867
N   31.536   58.941   39.931
C   31.773   57.728   39.265
C   33.194   57.481   39.161
C   33.848   58.628   39.684
C   32.749   59.560   40.137
C   33.748   56.097   38.749
C   35.333   58.758   39.811
O   36.026   57.942   39.275
C   36.114   59.950   40.477
N   30.618   61.524   40.966
C   31.942   61.798   41.031
C   32.206   63.297   41.333
C   30.717   63.909   41.482
C   29.821   62.637   41.350
C   33.105   63.403   42.621
C   30.467   65.059   40.413
C   29.748   66.340   40.988
N   27.869   60.544   40.762
C   27.474   61.811   41.227
C   26.019   61.903   41.395
C   25.617   60.608   40.874
C   26.753   59.856   40.513
C   25.194   63.140   41.987
C   24.517   59.734   40.808
O   23.357   59.910   41.240
C   24.993   58.452   40.012
C   24.627   57.265   40.771
O   25.111   56.930   41.860
O   23.694   56.637   40.005
C   23.284   55.262   40.375
C   26.150   54.786   33.593
C   26.821   55.930   32.878
C   27.339   55.846   31.613
C   27.237   54.602   30.805
C   28.087   57.022   31.091
C   29.562   56.863   30.695
C   29.771   56.694   29.191
C   30.713   55.491   28.829
C   30.420   54.976   27.414
C   32.200   55.877   29.038
C   32.913   55.513   30.374
C   33.832   56.631   30.875
C   35.313   56.385   30.456
C   36.290   57.073   31.448
C   35.624   56.629   28.938
C   35.616   55.369   28.149
C   36.863   55.287   27.175
C   37.538   53.935   27.334
C   38.007   53.664   28.829
C   38.644   53.695   26.233
H   31.086   55.735   38.663
H   33.952   61.259   40.669
H   28.206   63.780   41.787
H   26.521   55.625   39.657
H   28.877   55.272   37.884
H   29.550   53.922   39.321
H   27.883   53.672   39.986
H   28.980   54.672   40.868
H   27.024   56.482   36.778
H   25.527   56.905   37.501
H   24.375   54.960   37.067
H   25.731   54.163   37.925
H   34.381   56.141   37.863
H   33.007   55.381   38.395
H   34.295   55.715   39.611
H   36.146   60.709   39.695
H   37.153   59.707   40.700
H   35.684   60.442   41.349
H   32.660   63.631   40.400
H   30.614   64.234   42.517
H   33.195   62.452   43.147
H   32.634   64.156   43.254
H   34.092   63.823   42.429
H   29.811   64.656   39.641
H   31.423   65.319   39.958
H   30.553   67.069   41.079
H   29.288   66.105   41.948
H   29.061   66.715   40.229
H   24.500   62.981   42.812
H   24.655   63.559   41.137
H   25.917   63.875   42.340
H   24.589   58.487   39.001
H   23.029   54.648   39.511
H   22.531   55.338   41.159
H   24.202   54.799   40.738
H   25.292   54.367   33.066
H   26.904   53.999   33.597
H   27.049   56.864   33.392
H   26.917   54.784   29.778
H   26.427   53.939   31.110
H   28.151   54.031   30.644
H   28.077   57.699   31.945
H   27.521   57.370   30.227
H   29.931   56.022   31.282
H   30.148   57.674   31.127
H   30.167   57.653   28.857
H   28.804   56.568   28.703
H   30.530   54.617   29.454
H   30.527   53.892   27.422
H   31.004   55.493   26.652
H   29.402   55.170   27.075
H   32.248   56.949   28.845
H   32.744   55.328   28.270
H   33.478   54.647   30.029
H   32.375   55.159   31.254
H   33.807   56.552   31.962
H   33.419   57.560   30.483
H   35.331   55.318   30.678
H   36.758   56.319   32.082
H   35.772   57.715   32.160
H   37.023   57.734   30.987
H   36.548   57.185   28.779
H   34.839   57.214   28.459
H   34.777   55.355   27.454
H   35.515   54.449   28.725
H   37.658   55.990   27.427
H   36.463   55.551   26.197
H   36.711   53.241   27.178
H   39.037   53.309   28.856
H   37.478   52.855   29.331
H   38.021   54.562   29.447
H   39.521   53.213   26.666
H   38.968   54.594   25.707
H   38.228   52.946   25.559

