import { useEffect, useRef } from 'react'
import { X, Tag, Calendar, Database } from 'lucide-react'

interface Label {
  id: number
  source: string
  task: string
  data: {
    boxes?: number[][]
    class_names?: string[]
    confidences?: number[]
    coco_annotations?: any[]
  }
  created_at: string
}

interface ImageDetail {
  image: {
    id: number
    filename: string
    width: number
    height: number
    status: string
    dataset: string
    created_at: string
  }
  labels: Label[]
}

interface ImageModalProps {
  imageDetail: ImageDetail | null
  onClose: () => void
}

const COLORS = [
  '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
  '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9'
]

export default function ImageModal({ imageDetail, onClose }: ImageModalProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const imageRef = useRef<HTMLImageElement>(null)

  useEffect(() => {
    if (!imageDetail || !canvasRef.current || !imageRef.current) return

    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const img = imageRef.current

    const drawBoxes = () => {
      canvas.width = img.naturalWidth
      canvas.height = img.naturalHeight

      // Draw image
      ctx.drawImage(img, 0, 0)

      // Draw bounding boxes from labels
      imageDetail.labels.forEach((label) => {
        const boxes = label.data.boxes || []
        const classNames = label.data.class_names || []
        const confidences = label.data.confidences || []

        boxes.forEach((box, i) => {
          const [x1, y1, x2, y2] = box
          const color = COLORS[i % COLORS.length]
          const className = classNames[i] || 'object'
          const conf = confidences[i]

          // Draw rectangle
          ctx.strokeStyle = color
          ctx.lineWidth = 3
          ctx.strokeRect(x1, y1, x2 - x1, y2 - y1)

          // Draw label background
          const labelText = conf ? `${className} ${(conf * 100).toFixed(0)}%` : className
          ctx.font = 'bold 14px sans-serif'
          const textWidth = ctx.measureText(labelText).width

          ctx.fillStyle = color
          ctx.fillRect(x1, y1 - 22, textWidth + 8, 22)

          // Draw label text
          ctx.fillStyle = 'white'
          ctx.fillText(labelText, x1 + 4, y1 - 6)
        })
      })
    }

    if (img.complete) {
      drawBoxes()
    } else {
      img.onload = drawBoxes
    }
  }, [imageDetail])

  if (!imageDetail) return null

  const { image, labels } = imageDetail
  const totalBoxes = labels.reduce((sum, l) => sum + (l.data.boxes?.length || 0), 0)

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70" onClick={onClose}>
      <div
        className="bg-white rounded-xl shadow-2xl max-w-6xl max-h-[90vh] w-full mx-4 overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-200">
          <div>
            <h2 className="text-lg font-semibold text-slate-900">{image.filename}</h2>
            <p className="text-sm text-slate-500">
              {image.width}x{image.height} | {totalBoxes} detections
            </p>
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-slate-100 rounded-lg transition-colors"
          >
            <X className="h-5 w-5 text-slate-500" />
          </button>
        </div>

        {/* Content */}
        <div className="flex flex-col lg:flex-row">
          {/* Image with canvas overlay */}
          <div className="flex-1 p-4 bg-slate-900 flex items-center justify-center relative">
            <img
              ref={imageRef}
              src={`http://localhost:8000/api/images/${image.id}/file`}
              alt={image.filename}
              className="hidden"
              crossOrigin="anonymous"
            />
            <canvas
              ref={canvasRef}
              className="max-w-full max-h-[60vh] object-contain"
            />
            {labels.length === 0 && (
              <div className="absolute inset-0 flex items-center justify-center">
                <img
                  src={`http://localhost:8000/api/images/${image.id}/file`}
                  alt={image.filename}
                  className="max-w-full max-h-[60vh] object-contain"
                />
              </div>
            )}
          </div>

          {/* Sidebar */}
          <div className="w-full lg:w-80 border-t lg:border-t-0 lg:border-l border-slate-200 p-4 overflow-y-auto max-h-[60vh]">
            <h3 className="text-sm font-semibold text-slate-900 mb-3">Image Info</h3>

            <div className="space-y-3 mb-6">
              <div className="flex items-center gap-2 text-sm">
                <Database className="h-4 w-4 text-slate-400" />
                <span className="text-slate-600">Dataset:</span>
                <span className="text-slate-900">{image.dataset || 'N/A'}</span>
              </div>
              <div className="flex items-center gap-2 text-sm">
                <Calendar className="h-4 w-4 text-slate-400" />
                <span className="text-slate-600">Added:</span>
                <span className="text-slate-900">
                  {new Date(image.created_at).toLocaleDateString()}
                </span>
              </div>
            </div>

            <h3 className="text-sm font-semibold text-slate-900 mb-3">
              Detections ({totalBoxes})
            </h3>

            {labels.length === 0 ? (
              <p className="text-sm text-slate-500">No labels yet. Click auto-label to generate.</p>
            ) : (
              <div className="space-y-2">
                {labels.map((label) =>
                  (label.data.class_names || []).map((cls, i) => (
                    <div
                      key={`${label.id}-${i}`}
                      className="flex items-center justify-between p-2 bg-slate-50 rounded-lg"
                    >
                      <div className="flex items-center gap-2">
                        <div
                          className="w-3 h-3 rounded"
                          style={{ backgroundColor: COLORS[i % COLORS.length] }}
                        />
                        <span className="text-sm text-slate-700">{cls}</span>
                      </div>
                      {label.data.confidences?.[i] && (
                        <span className="text-xs text-slate-500">
                          {(label.data.confidences[i] * 100).toFixed(0)}%
                        </span>
                      )}
                    </div>
                  ))
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
