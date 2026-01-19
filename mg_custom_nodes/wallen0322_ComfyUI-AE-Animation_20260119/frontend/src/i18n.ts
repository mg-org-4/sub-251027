// Simplified i18n for mask editor
const translations: Record<string, string> = {
  'g.save': 'Save',
  'g.saving': 'Saving...',
  'g.cancel': 'Cancel',
  'g.undo': 'Undo',
  'g.redo': 'Redo',
  'maskeditor.invert': 'Invert',
  'maskeditor.clear': 'Clear'
}

export function t(key: string): string {
  return translations[key] || key
}

export function $t(key: string): string {
  return t(key)
}
