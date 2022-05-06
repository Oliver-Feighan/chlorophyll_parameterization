%nproc=24
%mem=50gb
#p ZINDO=(nstates=5)

step_1851_chromophore_25 ZINDO

0 1
Mg   -2.765   34.446   26.333
C   -3.982   32.679   29.011
C   -1.578   36.807   28.593
C   -2.194   36.469   23.789
C   -4.545   32.276   24.268
N   -2.967   34.807   28.590
C   -3.272   33.834   29.472
C   -2.918   34.106   30.968
C   -2.073   35.453   30.770
C   -2.213   35.750   29.198
C   -2.494   36.641   31.715
C   -2.059   32.893   31.564
C   -2.309   32.654   33.039
C   -1.203   32.592   34.007
O   -0.254   31.821   33.917
O   -1.454   33.512   35.001
N   -1.910   36.350   26.227
C   -1.432   37.169   27.208
C   -0.781   38.263   26.582
C   -0.920   38.206   25.143
C   -1.689   36.980   24.990
C   -0.048   39.317   27.401
C   -0.280   39.120   24.114
O   -0.378   38.976   22.907
C   0.729   40.140   24.494
N   -3.422   34.424   24.314
C   -2.902   35.374   23.480
C   -3.266   34.938   22.068
C   -3.655   33.475   22.198
C   -3.871   33.326   23.682
C   -4.329   35.817   21.423
C   -2.619   32.481   21.495
C   -1.305   32.340   22.188
N   -3.946   32.771   26.549
C   -4.611   31.978   25.617
C   -5.284   30.925   26.245
C   -5.102   31.131   27.615
C   -4.254   32.237   27.733
C   -5.987   29.789   25.610
C   -5.548   30.726   28.944
O   -6.373   29.883   29.317
C   -4.842   31.684   29.942
C   -5.820   32.230   30.892
O   -6.947   32.650   30.553
O   -5.359   32.274   32.166
C   -6.438   32.488   33.150
C   -0.699   33.503   36.146
C   -1.513   34.213   37.279
C   -1.375   33.915   38.565
C   -0.330   32.864   39.067
C   -2.332   34.477   39.593
C   -1.918   35.922   40.038
C   -1.862   36.118   41.535
C   -0.513   36.721   42.038
C   0.710   35.730   42.164
C   -0.735   37.547   43.361
C   -1.505   36.679   44.427
C   -0.636   36.575   45.777
C   0.176   35.303   46.157
C   -0.305   34.706   47.418
C   1.716   35.590   46.223
C   2.561   34.256   46.400
C   4.036   34.477   46.122
C   4.517   33.757   44.814
C   5.907   33.128   44.915
C   4.524   34.733   43.601
H   -1.114   37.477   29.320
H   -1.951   37.143   22.965
H   -4.989   31.564   23.568
H   -3.808   34.263   31.577
H   -1.072   35.133   31.061
H   -2.983   37.432   31.148
H   -1.521   36.828   32.171
H   -3.198   36.144   32.382
H   -0.989   32.996   31.386
H   -2.359   31.987   31.039
H   -2.770   31.668   33.102
H   -3.081   33.343   33.384
H   -0.311   40.326   27.084
H   1.000   39.112   27.183
H   -0.239   39.389   28.472
H   1.360   40.497   23.680
H   1.320   39.614   25.243
H   0.206   40.914   25.056
H   -2.355   35.041   21.478
H   -4.570   33.299   21.633
H   -3.823   36.646   20.929
H   -5.132   36.178   22.066
H   -4.791   35.240   20.623
H   -2.369   32.659   20.449
H   -3.131   31.539   21.691
H   -0.531   32.406   21.424
H   -1.334   31.410   22.755
H   -1.334   33.208   22.847
H   -6.003   28.979   26.339
H   -5.677   29.669   24.572
H   -7.062   29.958   25.547
H   -4.115   31.123   30.530
H   -6.951   31.527   33.152
H   -7.097   33.325   32.918
H   -6.054   32.506   34.170
H   0.268   33.944   35.907
H   -0.501   32.475   36.449
H   -2.329   34.894   37.037
H   0.699   33.099   38.796
H   -0.614   31.869   38.723
H   -0.444   32.737   40.143
H   -2.513   33.681   40.315
H   -3.303   34.516   39.100
H   -2.598   36.612   39.537
H   -0.938   36.099   39.595
H   -2.059   35.144   41.982
H   -2.590   36.873   41.831
H   -0.133   37.427   41.300
H   1.587   36.341   41.949
H   0.807   34.904   41.460
H   0.587   35.357   43.181
H   -1.276   38.462   43.119
H   0.277   37.823   43.658
H   -1.662   35.629   44.177
H   -2.448   37.213   44.549
H   -1.309   36.816   46.600
H   -0.011   37.469   45.779
H   -0.064   34.521   45.436
H   0.154   33.743   47.646
H   -1.374   34.506   47.354
H   -0.144   35.342   48.289
H   1.700   36.307   47.044
H   2.101   36.086   45.332
H   2.176   33.394   45.854
H   2.462   33.979   47.450
H   4.582   34.087   46.981
H   4.129   35.560   46.040
H   3.764   32.996   44.608
H   6.167   32.915   45.952
H   6.698   33.719   44.453
H   5.811   32.167   44.410
H   5.395   35.387   43.588
H   3.610   35.308   43.752
H   4.425   34.237   42.635

