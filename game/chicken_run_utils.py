import pygame
import sys


def load():
    PLAYER_PATH = (
            'assets/sprites/logo_small.png',
            'assets/sprites/logo_small.png',
            'assets/sprites/logo_small.png'
    )

    BACKGROUND_PATH = 'assets/sprites/background-white.png'

    BARRIER_PATH = (
        'assets/sprites/donuts_1.jpeg',
        'assets/sprites/donuts_2.jpeg',
        'assets/sprites/donuts_3.jpeg'
    )


    IMAGES, SOUNDS, HITMASKS = {}, {}, {}

    IMAGES['numbers'] = (
        pygame.image.load('assets/sprites/0.png').convert_alpha(),
        pygame.image.load('assets/sprites/1.png').convert_alpha(),
        pygame.image.load('assets/sprites/2.png').convert_alpha(),
        pygame.image.load('assets/sprites/3.png').convert_alpha(),
        pygame.image.load('assets/sprites/4.png').convert_alpha(),
        pygame.image.load('assets/sprites/5.png').convert_alpha(),
        pygame.image.load('assets/sprites/6.png').convert_alpha(),
        pygame.image.load('assets/sprites/7.png').convert_alpha(),
        pygame.image.load('assets/sprites/8.png').convert_alpha(),
        pygame.image.load('assets/sprites/9.png').convert_alpha()
    )

    IMAGES['base'] = pygame.image.load('assets/sprites/road.jpeg').convert_alpha()

    # sounds
    if 'win' in sys.platform:
        soundExt = '.wav'
    else:
        soundExt = '.ogg'

    SOUNDS['die']    = pygame.mixer.Sound('assets/audio/die' + soundExt)
    SOUNDS['hit']    = pygame.mixer.Sound('assets/audio/hit' + soundExt)
    SOUNDS['point']  = pygame.mixer.Sound('assets/audio/point' + soundExt)
    SOUNDS['swoosh'] = pygame.mixer.Sound('assets/audio/swoosh' + soundExt)
    SOUNDS['wing']   = pygame.mixer.Sound('assets/audio/wing' + soundExt)

    IMAGES['background'] = pygame.image.load(BACKGROUND_PATH).convert()

    IMAGES['player'] = (
        pygame.image.load(PLAYER_PATH[0]).convert_alpha(),
        pygame.image.load(PLAYER_PATH[1]).convert_alpha(),
        pygame.image.load(PLAYER_PATH[2]).convert_alpha(),
    )

    IMAGES['barrier'] = (
        pygame.image.load(BARRIER_PATH[0]).convert_alpha(),
        pygame.image.load(BARRIER_PATH[1]).convert_alpha(),
        pygame.image.load(BARRIER_PATH[2]).convert_alpha(),
    )

    HITMASKS['player'] = (
        getHitmask(IMAGES['player'][0]),
        getHitmask(IMAGES['player'][1]),
        getHitmask(IMAGES['player'][2]),
    )

    HITMASKS['barrier'] = (
        getHitmask(IMAGES['barrier'][0]),
        getHitmask(IMAGES['barrier'][1]),
        getHitmask(IMAGES['barrier'][2]),
    )

    return IMAGES, SOUNDS, HITMASKS


def getHitmask(image):
    '''
    :param
        an image

    :return:
        the hitmask of the image
    '''

    mask = []
    for x in range(image.get_width()):
        mask.append([])
        for y in range(image.get_height()):
            mask[x].append(bool(image.get_at((x,y))[3]))
    return mask
