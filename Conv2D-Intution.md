# Conv2D Intution
## #  3 x 3 kernel

### MODEL Having 1 Layer of Conv2D
```
model= keras.Sequential()
model.add(Conv2D(input_shape=(height, width, 1),
                 filters=64,
                 kernel_size=(3, 3))
        )
model.summary()
  ```

### Access Layer Parameter
```
  filters, _ = model.layers[0].get_weights()
  print("Filters Shape : ", filters.shape)
```

### Showing All The 64 unique Filters 
## Each Image is 3x3 **FILTER**
![](https://user-images.githubusercontent.com/74103314/223745730-7bc0b375-9c17-4be4-b706-b8b70262bf08.png)
