import sys
import pygame

sys.path.append("..")
from data_structure import ds_21card as ds
import visualize.vis as vis

banker1 =  ds.Banker(['CT','C3','C3','C4'],20)
player1 =  ds.Player(['C2','C6'],8)
player2 =  ds.Player(['C3','C7'],10)
player3 =  ds.Player(['H3','H8','S5'],16)

state1 = ds.StateResult(banker1,[player1])
state2 = ds.StateResult(banker1,[player1,player2])
state3 = ds.StateResult(banker1,[player1,player2,player3])

screen_image = pygame.display.set_mode((800,600))
pygame.display.set_caption('Blackjackist')
screen_image.fill(vis.bgcolor1)


while True:
    for event in pygame.event.get():
        if(event.type == pygame.QUIT):
            pygame.quit()
            sys.exit()
        vis.render(state1,screen_image)
        screen_image.fill(vis.bgcolor1)
        vis.render(state2,screen_image)
        screen_image.fill(vis.bgcolor1)
        vis.render(state3,screen_image)
        screen_image.fill(vis.bgcolor1)
        break
