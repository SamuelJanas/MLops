# Torch Lighning project

goal: use torch lightning, something for hyperparameters serch and something for logging. 

| lib | why |
| --- | --- | 
| lightning | Forced... |
| MLFlow | Never used it before wanted to take a look |
| optuna | Simple (but i kinda dislike it) |

---
MLFlow

Overall works nicely and easy enought syntax:

```sh
mlflow ui --backend-store-uri ./mlruns
```
Will start a UI based on local scores. Seems cool for self-hosting. I saw that they allow easy connection with databricks.

![dashboard metrics](scr/metrics.png)

The UI is incredibly similar to wandb and you can click through everything and it looks decent.

___
Lightning

Less boilerplate compared to raw torch. Great for when you don't need to control everything (which would be most of the times). 

There's still a little learning curve to the API, I don't see it replacing torch in older projects but it's nice not having to define everything.

___
Optuna

I've used it heavily in some other project and didn't really enjoy it. The dashboard is a little buggy. 
