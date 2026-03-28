package main

import (
    "fmt"
    "log"

	"github.com/go-chi/chi/v5"
    "github.com/go-chi/chi/v5/middleware"
    "github.com/berkaybgk/cv-urban-accessibility-senior-project/internal/config"
    "github.com/berkaybgk/cv-urban-accessibility-senior-project/internal/handler"
)

func main() {
    cfg, err := config.Load()
    if err != nil {
        // log.Fatal prints the error and calls os.Exit(1)
        // This is the standard way to abort at startup in Go
        log.Fatal("failed to load config: ", err)
    }

    ctx := context.Background()
	_ = ctx

	r := chi.NewRouter()
	r.Use(middleware.Logger)
	r.Use(middleware.Recoverer)

	r.Mount("/api", handler.NewRouter())
	log.Printf("server starting on port %s", cfg.Port)

	if err := http.ListenAndServe(cfg.Port, r); err != nil {
		log.Fatal("failed to start server: ", err)
	}
	
}
