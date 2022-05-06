%nproc=24
%mem=75GB
#p PBE1PBE/Def2SVP td=(nstates=5)

step_1801_chromophore_27 TDDFT with PBE1PBE functional

0 1
Mg   -5.024   24.809   26.707
C   -3.764   26.816   29.447
C   -6.013   22.564   29.183
C   -6.161   22.889   24.162
C   -3.399   26.844   24.504
N   -4.908   24.756   29.108
C   -4.516   25.779   29.979
C   -5.077   25.526   31.381
C   -5.575   24.009   31.310
C   -5.561   23.779   29.824
C   -4.614   23.014   32.117
C   -6.192   26.575   31.849
C   -5.976   27.221   33.215
C   -6.963   26.804   34.256
O   -7.784   27.481   34.810
O   -6.643   25.523   34.676
N   -5.960   22.970   26.644
C   -6.295   22.244   27.770
C   -6.836   21.023   27.265
C   -6.835   21.042   25.849
C   -6.357   22.357   25.450
C   -7.428   20.072   28.137
C   -7.308   20.072   24.751
O   -7.389   20.342   23.532
C   -7.594   18.643   25.224
N   -4.990   24.854   24.654
C   -5.549   24.067   23.741
C   -5.246   24.351   22.317
C   -4.328   25.668   22.436
C   -4.248   25.805   23.957
C   -4.560   23.172   21.570
C   -4.886   26.967   21.753
C   -3.991   27.713   20.798
N   -3.914   26.534   26.863
C   -3.279   27.202   25.857
C   -2.556   28.265   26.410
C   -2.684   28.138   27.774
C   -3.546   27.089   28.069
C   -1.715   29.263   25.552
C   -2.372   28.774   29.020
O   -1.630   29.725   29.251
C   -3.021   27.924   30.206
C   -1.887   27.379   31.065
O   -0.777   27.089   30.632
O   -2.279   27.213   32.354
C   -1.264   26.724   33.320
C   -7.100   25.160   36.042
C   -8.037   23.991   36.044
C   -8.266   23.012   36.925
C   -7.675   22.975   38.381
C   -9.006   21.745   36.550
C   -10.512   21.814   36.508
C   -11.337   21.077   37.572
C   -12.297   19.978   37.024
C   -13.709   20.379   36.779
C   -12.269   18.683   37.754
C   -11.588   17.506   37.140
C   -10.835   16.629   38.108
C   -9.293   16.896   38.123
C   -8.909   18.032   39.031
C   -8.392   15.653   38.497
C   -6.984   15.632   37.885
C   -6.018   15.075   38.993
C   -4.651   14.564   38.416
C   -3.857   13.863   39.594
C   -3.729   15.687   37.784
H   -6.303   21.810   29.918
H   -6.404   22.269   23.297
H   -2.767   27.383   23.794
H   -4.282   25.498   32.127
H   -6.563   23.713   31.662
H   -4.183   22.209   31.521
H   -5.190   22.582   32.935
H   -3.802   23.593   32.558
H   -7.179   26.111   31.849
H   -6.221   27.362   31.095
H   -5.982   28.297   33.041
H   -5.023   26.898   33.634
H   -8.443   19.898   27.779
H   -7.453   20.306   29.201
H   -6.870   19.149   27.979
H   -7.862   17.880   24.493
H   -8.391   18.717   25.963
H   -6.628   18.434   25.684
H   -6.253   24.476   21.920
H   -3.332   25.365   22.115
H   -4.003   22.557   22.277
H   -3.925   23.416   20.719
H   -5.370   22.494   21.299
H   -5.119   27.691   22.534
H   -5.816   26.817   21.205
H   -3.600   28.615   21.269
H   -4.472   27.904   19.839
H   -3.153   27.077   20.516
H   -2.208   29.811   24.750
H   -0.818   28.749   25.206
H   -1.479   29.919   26.390
H   -3.688   28.510   30.838
H   -0.310   27.250   33.277
H   -1.229   25.664   33.069
H   -1.764   26.852   34.279
H   -7.668   25.922   36.576
H   -6.146   24.955   36.528
H   -8.490   23.831   35.066
H   -7.102   22.063   38.550
H   -8.547   23.194   38.997
H   -6.999   23.759   38.722
H   -8.732   20.989   37.286
H   -8.651   21.346   35.600
H   -10.867   21.483   35.532
H   -10.802   22.846   36.309
H   -11.938   21.761   38.171
H   -10.630   20.673   38.297
H   -11.936   19.834   36.005
H   -13.874   21.291   37.352
H   -14.547   19.758   37.095
H   -13.964   20.543   35.732
H   -13.283   18.314   37.907
H   -11.984   18.804   38.799
H   -10.998   17.911   36.318
H   -12.171   16.803   36.545
H   -11.039   15.579   37.898
H   -11.323   16.717   39.079
H   -9.141   17.249   37.103
H   -9.794   18.266   39.623
H   -8.546   18.829   38.383
H   -8.128   17.786   39.751
H   -8.895   14.803   38.037
H   -8.375   15.537   39.580
H   -6.709   16.639   37.574
H   -6.859   14.950   37.044
H   -6.412   14.430   39.779
H   -5.659   15.986   39.473
H   -4.794   13.820   37.632
H   -4.411   14.156   40.486
H   -2.794   14.097   39.657
H   -3.972   12.791   39.439
H   -3.377   15.197   36.876
H   -2.865   15.930   38.402
H   -4.325   16.588   37.637

