.PHONY: up down logs ps

up:
	docker compose up -d --wait --wait-timeout 60

down:
	docker compose down

logs:
	docker compose logs --follow --tail=200

ps:
	docker compose ps
