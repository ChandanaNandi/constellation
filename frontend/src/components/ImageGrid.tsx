import { Eye, Tag, Loader2 } from 'lucide-react'

interface Image {
  id: number
  filename: string
  file_path: string
  width: number
  height: number
  status: string
  dataset: string
  created_at: string
}

interface ImageGridProps {
  images: Image[]
  onImageClick: (image: Image) => void
  onAutoLabel?: (imageId: number) => void
  isLabeling?: number | null
}

export default function ImageGrid({ images, onImageClick, onAutoLabel, isLabeling }: ImageGridProps) {
  const getImageUrl = (id: number) => `http://localhost:8000/api/images/${id}/file`

  return (
    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-4">
      {images.map((image) => (
        <div
          key={image.id}
          className="group relative bg-white rounded-lg shadow-sm border border-slate-200 overflow-hidden hover:shadow-md transition-shadow"
        >
          {/* Image */}
          <div className="aspect-video bg-slate-100 relative">
            <img
              src={getImageUrl(image.id)}
              alt={image.filename}
              className="w-full h-full object-cover"
              loading="lazy"
            />

            {/* Hover overlay */}
            <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center gap-2">
              <button
                onClick={() => onImageClick(image)}
                className="p-2 bg-white rounded-full hover:bg-slate-100 transition-colors"
                title="View details"
              >
                <Eye className="h-5 w-5 text-slate-700" />
              </button>
              {onAutoLabel && (
                <button
                  onClick={() => onAutoLabel(image.id)}
                  disabled={isLabeling === image.id}
                  className="p-2 bg-constellation-600 rounded-full hover:bg-constellation-700 transition-colors disabled:opacity-50"
                  title="Auto-label"
                >
                  {isLabeling === image.id ? (
                    <Loader2 className="h-5 w-5 text-white animate-spin" />
                  ) : (
                    <Tag className="h-5 w-5 text-white" />
                  )}
                </button>
              )}
            </div>
          </div>

          {/* Info */}
          <div className="p-2">
            <p className="text-xs text-slate-600 truncate" title={image.filename}>
              {image.filename}
            </p>
            <div className="mt-1">
              <span className="inline-flex text-xs px-1.5 py-0.5 rounded bg-slate-100 text-slate-700">
                {image.dataset || 'unknown'}
              </span>
            </div>
            <div className="flex items-center justify-between mt-1">
              <span className="text-xs text-slate-400">
                {image.width}x{image.height}
              </span>
              <span className={`text-xs px-1.5 py-0.5 rounded ${
                image.status === 'labeled'
                  ? 'bg-green-100 text-green-700'
                  : 'bg-yellow-100 text-yellow-700'
              }`}>
                {image.status}
              </span>
            </div>
          </div>
        </div>
      ))}
    </div>
  )
}
