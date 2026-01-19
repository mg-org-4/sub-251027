import { createApp, type App as VueApp } from 'vue'
import { createPinia } from 'pinia'
import { app } from "../../../scripts/app.js"
import PrimeVue from 'primevue/config'
import Root from "@/Root.vue"
import { i18n } from "@/i18n"
import './style.css'

const { ComfyButton } = window.comfyAPI.button

let vueApp: VueApp | null = null
let mountContainer: HTMLElement | null = null
let rootInstance: InstanceType<typeof Root> | null = null

function ensureOpenCutInstance(): InstanceType<typeof Root> {
    if (mountContainer && rootInstance) {
        return rootInstance
    }

    mountContainer = document.createElement('div')
    mountContainer.id = 'opencut-root'
    document.body.appendChild(mountContainer)

    vueApp = createApp(Root)
    vueApp.use(createPinia())
    vueApp.use(i18n)
    vueApp.use(PrimeVue, {
        theme: 'none'
    })
    rootInstance = vueApp.mount(mountContainer) as InstanceType<typeof Root>

    return rootInstance
}

function openOpenCut() {
    const instance = ensureOpenCutInstance()
    instance.open()
}

app.registerExtension({
    name: 'ComfyUI.OpenCut.TopMenu',
    setup() {
        app.menu?.settingsGroup.append(
            new ComfyButton({
                icon: 'video',
                tooltip: 'comfyui-opencut',
                content: 'OpenCut',
                action: openOpenCut,
            }),
        )
    },
})
