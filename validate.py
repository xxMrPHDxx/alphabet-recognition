from tensorflow import convert_to_tensor
from model import Model
import pygame

if __name__ == '__main__':
	model = Model()
	model.load_weights('model')

	pygame.init()
	screen = pygame.display.set_mode((400, 400))
	clock  = pygame.time.Clock()
	FPS    = 30

	WHITE       = (255, 255, 255)
	BLACK       = (0, 0, 0)
	RED         = (255, 0, 0)
	PIXEL_SCALE = 255*3
	ALPHABETS   = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

	buffer = pygame.Surface((400, 400))
	buffer.fill(WHITE)

	drawing = False
	guess   = None
	lx, ly  = None, None
	font    = pygame.font.Font('freesansbold.ttf', 16)

	running = True
	while running:
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				running = False
			if event.type == pygame.MOUSEBUTTONDOWN:
				drawing = True
			if event.type == pygame.MOUSEBUTTONUP:
				drawing = False
			if event.type == pygame.KEYDOWN and event.key == ord('c'):
				buffer.fill(WHITE)

		# Drawing with the mouse
		if drawing:
			x, y = pygame.mouse.get_pos()
			if lx != None:
				pygame.draw.line(buffer, BLACK, (lx, ly), (x, y), 24)
			lx, ly = x, y
		else:
			lx, ly = None, None

		# Copy current screen to buffer for guessing
		scaled = pygame.transform.scale(buffer, (28, 28))

		# The guessing begins
		pixels = pygame.PixelArray(scaled)
		inputs = []
		for y in range(28):
			arr = []
			for x in range(28):
				col = pixels[x, y]
				r = (col >> 16) & 0xff
				g = (col >>  8) & 0xff
				b = (col >>  0) & 0xff
				arr.append([1.0 - sum([r, g, b]) / PIXEL_SCALE])
			inputs.append(arr)
		del pixels
		inputs = convert_to_tensor([inputs])
		guess  = model.predict(inputs)[0, :]
		_max   = max(guess)
		guess  = ALPHABETS[[
			i 
			for i, val in enumerate(guess) 
			if val == _max
		][0]]

		screen.blit(buffer, (0, 0))
		screen.blit(
			font.render(f'Guess: {guess}', True, RED, WHITE), 
			(10, 10)
		)

		# TODO: For debugging
		screen.blit(scaled, (400-28, 0))

		# Update display and simulate a tick
		pygame.display.update()
		clock.tick(FPS)
