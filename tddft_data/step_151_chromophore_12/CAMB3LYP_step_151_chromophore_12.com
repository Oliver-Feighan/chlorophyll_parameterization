%nproc=24
%mem=75GB
#p cam-b3lyp/Def2SVP td=(nstates=5)

step_151_chromophore_12 TDDFT with cam-b3lyp functional

0 1
Mg   48.240   15.222   28.100
C   46.109   14.965   30.761
C   50.301   17.273   29.954
C   50.003   15.874   25.273
C   45.822   13.670   26.010
N   48.232   16.032   30.143
C   47.249   15.726   31.080
C   47.656   16.416   32.451
C   48.725   17.458   31.980
C   49.186   16.822   30.666
C   48.151   18.882   31.765
C   48.213   15.318   33.500
C   47.698   15.524   34.954
C   48.700   15.614   36.075
O   49.338   14.582   36.408
O   48.906   16.846   36.641
N   50.036   16.392   27.730
C   50.748   17.108   28.638
C   51.856   17.658   27.920
C   51.778   17.347   26.538
C   50.551   16.495   26.457
C   52.931   18.441   28.647
C   52.685   17.790   25.399
O   52.419   17.533   24.234
C   53.944   18.568   25.684
N   48.005   14.859   25.965
C   48.860   15.154   25.040
C   48.641   14.367   23.731
C   47.286   13.709   23.974
C   47.035   14.062   25.429
C   48.871   15.158   22.375
C   47.194   12.290   23.534
C   46.200   12.060   22.402
N   46.338   14.569   28.221
C   45.496   13.820   27.378
C   44.281   13.445   28.087
C   44.513   13.804   29.416
C   45.783   14.418   29.500
C   43.211   12.679   27.422
C   43.969   13.758   30.760
O   42.874   13.322   31.059
C   44.954   14.509   31.676
C   44.297   15.698   32.192
O   43.838   16.570   31.524
O   44.427   15.665   33.538
C   44.056   16.864   34.254
C   50.005   16.764   37.625
C   50.015   18.064   38.349
C   51.085   18.516   38.983
C   52.188   17.614   39.535
C   51.271   19.973   39.376
C   52.736   20.601   39.260
C   52.836   22.026   39.773
C   53.649   22.969   38.964
C   54.796   23.674   39.803
C   52.770   24.108   38.357
C   51.859   23.679   37.124
C   51.465   24.885   36.154
C   52.046   24.645   34.713
C   52.997   25.878   34.443
C   50.933   24.445   33.639
C   51.403   24.904   32.221
C   50.454   24.521   31.070
C   51.031   23.834   29.823
C   50.067   23.770   28.651
C   51.712   22.468   30.117
H   50.810   18.024   30.562
H   50.574   15.948   24.345
H   45.051   13.159   25.430
H   46.774   16.900   32.871
H   49.571   17.560   32.660
H   48.868   19.690   31.906
H   47.336   18.958   32.485
H   47.752   18.986   30.756
H   49.301   15.298   33.427
H   47.791   14.389   33.116
H   47.004   14.741   35.257
H   47.164   16.474   34.979
H   53.056   19.450   28.256
H   53.917   17.991   28.533
H   52.629   18.553   29.689
H   53.674   19.558   26.053
H   54.553   18.724   24.793
H   54.452   17.957   26.430
H   49.435   13.641   23.905
H   46.545   14.316   23.454
H   48.218   14.783   21.588
H   49.946   15.093   22.206
H   48.742   16.226   22.547
H   46.881   11.693   24.390
H   48.159   11.912   23.198
H   46.672   11.564   21.554
H   45.721   12.978   22.060
H   45.408   11.484   22.880
H   43.452   11.641   27.649
H   43.001   12.998   26.401
H   42.292   12.832   27.988
H   45.225   13.920   32.552
H   43.779   16.568   35.266
H   43.154   17.362   33.899
H   44.883   17.565   34.146
H   50.921   16.652   37.045
H   49.629   15.944   38.237
H   49.326   18.822   37.975
H   52.588   18.009   40.468
H   53.062   17.692   38.888
H   51.824   16.618   39.791
H   50.766   20.274   40.293
H   50.729   20.456   38.563
H   53.002   20.489   38.209
H   53.391   19.986   39.877
H   53.336   21.801   40.714
H   51.869   22.409   40.101
H   54.271   22.544   38.177
H   54.304   24.635   39.953
H   55.691   23.714   39.183
H   55.008   23.218   40.770
H   53.277   25.049   38.147
H   52.080   24.340   39.169
H   50.940   23.454   37.665
H   52.319   22.765   36.749
H   51.820   25.806   36.616
H   50.385   24.787   36.260
H   52.612   23.714   34.683
H   53.488   26.096   35.391
H   52.477   26.762   34.072
H   53.657   25.443   33.693
H   49.978   24.914   33.874
H   50.616   23.403   33.671
H   52.389   24.503   31.986
H   51.399   25.994   32.229
H   50.070   25.469   30.693
H   49.688   23.832   31.426
H   51.767   24.552   29.461
H   49.880   22.747   28.323
H   50.399   24.335   27.780
H   49.123   24.203   28.983
H   51.262   21.901   30.932
H   52.751   22.613   30.412
H   51.564   21.940   29.175

