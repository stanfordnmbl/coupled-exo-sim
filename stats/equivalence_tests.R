require("TOSTER")

citation()
citation("TOSTER")

bound = 5
print("Hip extension, knee extension")
TOSTpaired.raw(n=5, 
           m1=11.87, 
           m2=13.60, 
           sd1=2.26, 
           sd2=1.42, 
           r12=-0.36, 
           low_eqbound=-bound,
           high_eqbound=bound)

bound = 4
print("Hip flexion, knee flexion")
TOSTpaired.raw(n=5, 
           m1=25.34, 
           m2=29.52, 
           sd1=5.41, 
           sd2=4.22, 
           r12=0.90, 
           low_eqbound=-bound,
           high_eqbound=bound)

bound = 6
print("Knee flexion, ankle plantarflexion")
TOSTpaired.raw(n=5, 
           m1=29.69, 
           m2=31.67, 
           sd1=4.64, 
           sd2=3.29, 
           r12=0.99, 
           low_eqbound=-bound,
           high_eqbound=bound)

bound = 6
print("Hip flexion, ankle plantarflexion")
TOSTpaired.raw(n=5, 
           m1=29.17, 
           m2=33.56, 
           sd1=2.46, 
           sd2=4.92, 
           r12=0.96, 
           low_eqbound=-bound,
           high_eqbound=bound)

bound = 8
print("Hip flexion, knee flexion, ankle plantarflexion")
TOSTpaired.raw(n=5, 
           m1=34.19, 
           m2=39.39, 
           sd1=2.59, 
           sd2=6.94, 
           r12=0.84, 
           low_eqbound=-bound,
           high_eqbound=bound)