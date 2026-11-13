.PHONY: up down logs ps profile

up:
	docker compose up --build -d --wait --wait-timeout 60

down:
	docker compose down

logs:
	docker compose logs --follow --tail=200

ps:
	docker compose ps

profile:
	@test -n "$(P)" || (echo "Usage: make profile P=<fast|typical|slow|flaky>" >&2; exit 2)
	uv run pa-toxiproxy apply "$(P)"
