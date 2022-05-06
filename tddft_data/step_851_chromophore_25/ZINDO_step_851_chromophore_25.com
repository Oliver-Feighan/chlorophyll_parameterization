%nproc=24
%mem=50gb
#p ZINDO=(nstates=5)

step_851_chromophore_25 ZINDO

0 1
Mg   -3.338   34.658   26.186
C   -4.642   33.048   29.056
C   -2.185   37.139   28.223
C   -3.319   36.819   23.640
C   -4.969   32.338   24.219
N   -3.333   34.981   28.436
C   -3.910   34.190   29.367
C   -3.557   34.674   30.739
C   -2.838   36.091   30.450
C   -2.801   36.130   28.943
C   -3.670   37.269   31.149
C   -2.549   33.714   31.457
C   -2.941   33.112   32.773
C   -1.988   33.397   33.908
O   -0.930   32.742   34.117
O   -2.365   34.511   34.658
N   -2.499   36.561   25.897
C   -2.167   37.417   26.876
C   -1.680   38.633   26.279
C   -2.044   38.603   24.892
C   -2.660   37.313   24.758
C   -0.954   39.812   27.063
C   -1.852   39.658   23.826
O   -2.210   39.480   22.656
C   -1.462   40.998   24.310
N   -4.001   34.554   24.205
C   -3.898   35.623   23.344
C   -4.648   35.220   21.980
C   -4.709   33.684   22.024
C   -4.569   33.508   23.574
C   -6.015   35.932   21.896
C   -3.509   32.868   21.367
C   -3.885   31.815   20.361
N   -4.426   32.900   26.509
C   -4.960   32.049   25.600
C   -5.652   30.975   26.313
C   -5.535   31.304   27.666
C   -4.831   32.545   27.732
C   -6.313   29.762   25.707
C   -5.867   30.826   28.972
O   -6.480   29.816   29.338
C   -5.297   31.916   29.986
C   -6.317   32.429   30.933
O   -7.237   33.257   30.721
O   -6.119   31.710   32.184
C   -6.998   32.093   33.239
C   -1.491   34.883   35.788
C   -2.386   35.171   37.012
C   -2.691   34.336   38.041
C   -1.837   33.122   38.404
C   -3.887   34.544   38.877
C   -3.732   35.088   40.355
C   -2.417   35.898   40.624
C   -2.905   37.358   41.100
C   -2.774   38.329   39.857
C   -1.902   37.804   42.218
C   -2.275   37.233   43.613
C   -1.230   37.669   44.710
C   -0.217   36.471   44.946
C   -0.542   35.633   46.211
C   1.222   37.120   45.050
C   2.342   36.150   44.730
C   3.067   36.719   43.530
C   3.399   35.596   42.498
C   2.094   35.054   41.888
C   4.377   34.488   42.895
H   -1.698   37.833   28.911
H   -3.383   37.431   22.738
H   -5.540   31.642   23.601
H   -4.493   34.725   31.296
H   -1.773   36.041   30.676
H   -4.642   36.808   31.322
H   -3.812   38.063   30.416
H   -3.198   37.642   32.058
H   -1.590   34.216   31.587
H   -2.345   32.876   30.791
H   -3.030   32.031   32.668
H   -3.932   33.488   33.027
H   0.046   40.006   26.675
H   -0.803   39.488   28.093
H   -1.600   40.682   27.175
H   -2.072   41.257   25.176
H   -1.668   41.740   23.539
H   -0.428   41.030   24.653
H   -4.080   35.503   21.094
H   -5.608   33.239   21.598
H   -6.129   36.846   21.313
H   -6.320   36.158   22.918
H   -6.715   35.220   21.457
H   -2.972   32.267   22.102
H   -2.797   33.592   20.971
H   -3.920   30.806   20.772
H   -3.208   31.872   19.508
H   -4.899   31.992   20.003
H   -5.620   29.205   25.076
H   -7.226   30.033   25.176
H   -6.606   29.187   26.585
H   -4.495   31.417   30.530
H   -7.989   32.267   32.820
H   -6.563   32.818   33.927
H   -7.131   31.303   33.978
H   -0.875   35.765   35.613
H   -0.760   34.113   36.036
H   -3.035   36.042   37.104
H   -0.968   33.055   37.750
H   -2.444   32.224   38.279
H   -1.556   33.208   39.453
H   -4.283   33.543   39.049
H   -4.690   34.991   38.292
H   -3.751   34.385   41.188
H   -4.571   35.779   40.438
H   -1.838   35.949   39.702
H   -1.793   35.297   41.285
H   -3.895   37.397   41.555
H   -2.151   39.181   40.126
H   -3.715   38.753   39.506
H   -2.468   37.725   39.002
H   -1.920   38.880   42.393
H   -0.910   37.498   41.883
H   -2.301   36.144   43.588
H   -3.250   37.513   44.012
H   -1.687   37.976   45.650
H   -0.708   38.564   44.372
H   -0.146   35.807   44.085
H   -1.247   36.179   46.837
H   0.367   35.461   46.787
H   -1.070   34.711   45.970
H   1.348   37.349   46.108
H   1.291   38.053   44.491
H   1.990   35.131   44.568
H   3.146   36.102   45.464
H   4.069   36.919   43.908
H   2.669   37.628   43.079
H   3.862   36.172   41.697
H   1.213   35.684   42.007
H   1.890   34.126   42.422
H   2.262   34.749   40.855
H   4.587   34.529   43.963
H   5.330   34.467   42.366
H   3.980   33.476   42.809

