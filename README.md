## 📌 Description
run multiple statistic routines and generate a single report (html, pdf, md)
## 📸 Screenshot
![screenshot1](https://github.com/pemn/assets/blob/main/db_multi_stats1.png?raw=true)
## 📝 Parameters
name|optional|description
---|---|------
input_path|❎|path a supported file format with data
condition|☑️|only rows where the given path expression is True will be used
series|☑️|a classification variable to be used as series. ex.: lito
values|❎|numeric variables with data values. ex.: fegl, sigl
fivenum||create a table with standard descriptive statistics: mean,min,max,std
||weight|variable for using as mean weight
boxplot_lito||create a boxplot with lito as series
boxplot_variable||create a boxplot with variables as series
histogram||standard histogram for value variables
scatter||create pairs of scatter between each 2 of the value variables. ex.: fegl and fegl_nn
output|☑️|path to save a file with the final report.

## 📓 Notes
## 📚 Examples
![screenshot2](https://github.com/pemn/assets/blob/main/db_multi_stats2.png?raw=true)
## 🧩 Compatibility
distribution|status
---|---
![winpython_icon](https://github.com/pemn/assets/blob/main/winpython_icon.png?raw=true)|✔
![vulcan_icon](https://github.com/pemn/assets/blob/main/vulcan_icon.png?raw=true)|✔
![anaconda_icon](https://github.com/pemn/assets/blob/main/anaconda_icon.png?raw=true)|❌
## 🙋 Support
Any question or problem contact:
 - paulo.ernesto
## 💎 License
Apache 2.0
Copyright ![vale_logo_only](https://github.com/pemn/assets/blob/main/vale_logo_only_r.svg?raw=true) Vale 2023
