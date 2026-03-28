package config

import (
	"fmt"
	"os"

	// dotenv
	"github.com/joho/godotenv"
)

type Config struct {
	Port string
	DatabaseURL string
	RabbitMQURL string
	GCSBucket string
	GCPProjectID string
}

func Load() (*Config, error) {
	err := godotenv.Load()
	if err != nil {
		return nil, fmt.Errorf("error loading .env file: %w", err)
	}

	return &Config{
		Port: getEnv("PORT", "8080"),
		DatabaseURL: os.Getenv("DATABASE_URL"),
		RabbitMQURL: getEnv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/"),
		GCSBucket: getEnv("GCS_BUCKET", "cv-urban-accessibility-bucket"),
		GCPProjectID: getEnv("GCP_PROJECT_ID", "cv-urban-accessibility"),
	}, nil
}

func getEnv(key string, defaultValue string) string {
	value := os.Getenv(key)
	if value == "" {
		return defaultValue
	}
	return value
}
