library(terra)
library(ggplot2)

pre <- rast("/cloud/project/activity07/ldd_pre.tif")
post <- rast("/cloud/project/activity07/ldd_post.tif")

par(mfrow=c(1,2))

plotRGB(post, r=3,g=2,b=1, # image and bands
        scale=13000, # note high scale due to clouds
        stretch="lin", # contrast stretch
        main="True color",
        axes=TRUE)
plotRGB(post, r=4,g=3,b=2, # image and bands
        scale=13000,# max data value
        stretch="lin", # contrast stretch
        main="False color",
        axes=TRUE)

northBound <- vect("/cloud/project/activity07/north_bound.shp")
midBound <- vect("/cloud/project/activity07/mid_bound.shp")

plotRGB(post, r=3,g=2,b=1, stretch="lin")
terra::plot(northBound, border="tomato3", col=NA, lwd=3, add=TRUE)
terra::plot(midBound, border="royalblue4",col=NA, lwd=3, add=TRUE)
#terra::plot(pre[1], col=gray(1:100/100))
#plotRGB(pre, r=3, g=2, b=1, scale=13000, stretch="lin",axes=TRUE)
help(plot)

preR <- pre/10000
plot(preR[[1]])

#calculate NDVI pre from NIR (4th band) and Red (3rd band)
ndvi_pre <- (pre[[4]]-pre[[3]])/(pre[[4]]+pre[[3]])
#calculate NDVI post
ndvi_post <- (post[[4]]-post[[3]])/(post[[4]]+post[[3]])

par(mfrow=c(1,2))
terra::plot(ndvi_pre)
terra::plot(ndvi_post)


# difference between post and pre ndvi
ndviDiff <- ndvi_post - ndvi_pre
terra::plot(ndviDiff)

# return a vector of cell values from the first polygon
ndvi_North <- extract(ndviDiff,northBound)
# return a vector of cell values from the first polygon
ndvi_Mid <- extract(ndviDiff,midBound)

ndviDataframe <- data.frame(location=c(rep("North", length(ndvi_North[,2])),
                                       rep("Mid", length(ndvi_Mid[,2]))),
                            ndvi.diff =c(ndvi_North[,2],ndvi_Mid[,2]))


ggplot(data=ndviDataframe,
       aes(x=location,ndvi.diff))+
  geom_boxplot(outlier.shape = NA)+
  ylim(-0.5,0.5)

