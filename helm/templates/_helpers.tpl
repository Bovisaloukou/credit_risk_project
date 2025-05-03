{{/*
Expand the name of the chart.
*/}}
{{- define "credit-risk-predictor.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "credit-risk-predictor.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- $name := include "credit-risk-predictor.name" . -}}
{{- if contains $name .Release.Name -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{/*
Create chart directlabels to be used as selectors.
*/}}
{{- define "credit-risk-predictor.selectorLabels" -}}
app.kubernetes.io/name: {{ include "credit-risk-predictor.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{/*
Create the labels that are used in every object.
*/}}
{{- define "credit-risk-predictor.labels" -}}
{{- include "credit-risk-predictor.selectorLabels" . | nindent 4 }}
helm.sh/chart: {{ include "credit-risk-predictor.chart" . }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end -}}

{{/*
Chart version
*/}}
{{- define "credit-risk-predictor.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
Common labels
*/}}
{{- define "credit-risk-predictor.commonLabels" -}}
app: credit-risk-predictor
{{- end -}}

{{/*
Create the name of the service account to use
*/}}
{{- define "credit-risk-predictor.serviceAccountName" -}}
{{- if .Values.serviceAccount.create -}}
{{- default (include "credit-risk-predictor.fullname" .) .Values.serviceAccount.name -}}
{{- else -}}
{{- default "default" .Values.serviceAccount.name -}}
{{- end -}}
{{- end -}}