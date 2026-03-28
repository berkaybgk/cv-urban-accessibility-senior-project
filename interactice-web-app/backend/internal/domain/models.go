package domain

import (
    "encoding/json"
    "time"
)

type Point struct {
	ID string `json:"id"`
	Latitude float64 `json:"latitude"`
	Longitude float64 `json:"longitude"`
	Timestamp time.Time `json:"timestamp"`
}

type AttachedImage struct {
	ID string `json:"id"`
	PointID string `json:"point_id"`
	ImageURL string `json:"image_url"`
	ImageType string `json:"image_type"`
	ImageSize int64 `json:"image_size"`
	ImageWidth int `json:"image_width"`
	ImageHeight int `json:"image_height"`
}

type UserComment struct {
	ID string `json:"id"`
	AuthorName string `json:"author_name"`
	PointID string `json:"point_id"`
	Comment string `json:"comment"`
	Timestamp time.Time `json:"timestamp"`
	AttachedImageID *string `json:"attached_image_id,omitempty"`
	RatingID *string `json:"rating_id,omitempty"`
}

type Rating struct {
	ID string `json:"id"`
	PointID string `json:"point_id"`
	Score int `json:"score"`
	Timestamp time.Time `json:"timestamp"`
}

type EventType string

const (
    EventPinCreated     EventType = "pin.created"
    EventPinDeleted     EventType = "pin.deleted"
    EventCommentCreated EventType = "comment.created"
    EventRatingCreated  EventType = "rating.created"
    EventImageAttached  EventType = "image.attached"
)


type Event struct {
    ID        string          `json:"id"`
    Type      EventType       `json:"type"`
    Payload   json.RawMessage `json:"payload"`
    Timestamp time.Time       `json:"timestamp"`
}

