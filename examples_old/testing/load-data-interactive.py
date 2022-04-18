from cosmosim.core.animation import InteractiveAnimation
scale=6.5e-9
path = "C:/test_data/cosmosim/test_run/"

animation = InteractiveAnimation(path, scale=scale)
animation.play(paused=True)