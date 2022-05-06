%nproc=24
%mem=75GB
#p cam-b3lyp/Def2SVP td=(nstates=5)

step_251_chromophore_3 TDDFT with cam-b3lyp functional

0 1
Mg   1.277   7.976   26.813
C   1.705   10.259   29.275
C   1.827   5.414   29.093
C   0.855   5.693   24.349
C   1.218   10.518   24.464
N   1.593   7.920   28.894
C   1.775   8.962   29.744
C   1.816   8.528   31.186
C   2.131   6.987   31.088
C   1.797   6.748   29.579
C   3.535   6.424   31.564
C   0.414   8.754   31.912
C   0.558   8.896   33.456
C   1.718   8.244   34.227
O   2.862   8.582   34.327
O   1.373   7.036   34.796
N   1.404   5.819   26.673
C   1.637   4.964   27.782
C   1.599   3.600   27.348
C   1.328   3.646   25.934
C   1.171   5.112   25.624
C   1.722   2.420   28.247
C   1.184   2.499   24.940
O   1.059   2.620   23.686
C   1.207   1.067   25.447
N   1.136   8.072   24.713
C   0.771   6.994   23.905
C   0.285   7.387   22.480
C   0.389   9.008   22.619
C   0.908   9.244   23.990
C   1.234   6.776   21.441
C   -0.882   9.739   22.301
C   -0.724   10.776   21.226
N   1.524   9.916   26.737
C   1.454   10.879   25.738
C   1.659   12.168   26.287
C   1.814   12.015   27.691
C   1.620   10.641   27.946
C   1.673   13.482   25.605
C   2.151   12.694   28.923
O   2.455   13.832   29.223
C   2.007   11.526   30.010
C   3.064   11.590   31.053
O   4.238   11.793   30.876
O   2.591   11.432   32.299
C   3.633   11.477   33.364
C   2.310   6.247   35.623
C   1.622   5.366   36.586
C   1.576   5.249   37.938
C   2.528   5.957   38.893
C   0.744   4.272   38.617
C   1.506   2.858   38.766
C   0.715   1.763   38.177
C   0.467   0.587   39.184
C   -1.027   0.234   39.166
C   1.257   -0.656   38.869
C   2.660   -0.692   39.548
C   3.453   -1.877   38.991
C   4.216   -2.565   40.104
C   3.193   -3.388   41.037
C   5.089   -1.615   40.940
C   6.488   -2.136   41.104
C   7.468   -1.257   41.949
C   8.800   -0.782   41.272
C   9.808   -0.320   42.293
C   8.492   0.451   40.351
H   2.182   4.705   29.844
H   0.854   4.952   23.547
H   1.099   11.382   23.807
H   2.671   8.941   31.721
H   1.356   6.446   31.630
H   4.079   7.254   32.015
H   4.169   5.949   30.817
H   3.322   5.630   32.280
H   -0.184   7.894   31.610
H   -0.017   9.667   31.501
H   -0.392   8.497   33.812
H   0.495   9.963   33.668
H   0.812   2.124   28.769
H   2.522   2.672   28.942
H   2.186   1.560   27.765
H   2.260   0.802   25.541
H   0.757   0.481   24.646
H   0.719   0.929   26.412
H   -0.695   6.925   22.360
H   1.131   9.367   21.905
H   2.138   6.312   21.836
H   1.419   7.511   20.658
H   0.691   5.938   21.003
H   -1.275   10.306   23.145
H   -1.714   9.099   22.007
H   -1.487   10.709   20.451
H   0.240   10.686   20.726
H   -0.684   11.719   21.772
H   2.028   13.483   24.574
H   2.512   13.995   26.076
H   0.695   13.956   25.689
H   1.083   11.889   30.461
H   3.952   12.497   33.583
H   4.445   10.776   33.170
H   3.188   11.107   34.288
H   3.067   6.844   36.132
H   2.816   5.600   34.907
H   0.914   4.685   36.114
H   3.126   6.682   38.341
H   3.138   5.224   39.422
H   2.022   6.528   39.672
H   -0.241   4.151   38.168
H   0.465   4.663   39.595
H   1.872   2.806   39.792
H   2.388   3.046   38.153
H   1.292   1.604   37.266
H   -0.161   2.168   37.670
H   0.534   0.958   40.207
H   -1.305   -0.357   40.039
H   -1.351   -0.351   38.305
H   -1.732   1.051   39.316
H   1.398   -0.695   37.789
H   0.727   -1.541   39.220
H   2.456   -0.664   40.618
H   3.038   0.279   39.229
H   4.225   -1.543   38.298
H   2.883   -2.617   38.428
H   4.887   -3.251   39.585
H   3.074   -2.951   42.029
H   3.649   -4.376   41.091
H   2.207   -3.620   40.634
H   4.723   -1.568   41.966
H   5.027   -0.572   40.631
H   6.929   -2.403   40.143
H   6.353   -2.970   41.793
H   7.836   -1.781   42.831
H   6.948   -0.359   42.284
H   9.264   -1.596   40.713
H   10.227   0.669   42.109
H   10.540   -1.126   42.229
H   9.409   -0.297   43.307
H   7.444   0.401   40.057
H   9.033   0.214   39.435
H   8.707   1.454   40.721

