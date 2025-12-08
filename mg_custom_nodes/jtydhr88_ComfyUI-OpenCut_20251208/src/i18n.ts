import { createI18n } from 'vue-i18n'

const messages = {
  en: {
    opencut: {
      title: 'OpenCut'
    }
  },
  zh: {
    opencut: {
      title: 'OpenCut'
    }
  }
}

export const i18n = createI18n({
  legacy: false,
  locale: navigator.language.split('-')[0] || 'en',
  fallbackLocale: 'en',
  messages
})
